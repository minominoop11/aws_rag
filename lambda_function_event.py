'''
AWS IoT Core를 사용하여 AI 스마트 카메라 디바이스가 이벤트를 감지하고, 해당 이벤트에 대한 메타데이터를 게시하며, 클라우드로부터 Presigned URL을 받아 이미지를 업로드하는 과정을 구현합니다.

[구현순서]
1. MQTT 연결
2. MQTT 토픽 & 페이로드
 1) 이벤트 게시
    - 이벤트 게시 토픽: sites/{siteId}/cameras/{deviceId}/events
    - Presigned 응답 토픽: sites/{siteId}/cameras/{deviceId}/uploads/presign
    - 예시 페이로드 (카메라 → 클라우드)
    {
    "siteId": "OCTANK-1",
    "deviceId": "3F-07",
    "eventId": "542991",
    "ts": "2025-07-27T08:15:23Z",
    "eventType": "DANGER_ZONE_INTRUSION",
    "severity": "HIGH",
    "message": "작업자 A가 탱크구역 A 폴리곤 내부 진입",
    "roiId": "TANK-AREA-A",
    "model": { "name": "yolov8n", "ver": "1.3.2", "conf": 0.82 },
    "imageRequired": true
    }
 2) presigned 수신 대기
    - 예시 응답 (클라우드 → 카메라)
    {
    "eventId": "542991",
    "s3Bucket": "sgai-raw-images",
    "s3Key": "OCTANK-1/3F-07/2025/07/27/542991.jpg",
    "url": "<presigned_put_url>",
    "headers": { "Content-Type": "image/jpeg" },
    "expireSec": 300
    }
    - S3 키 규칙
    raw:    s3://sgai-raw-images/{siteId}/{deviceId}/{YYYY}/{MM}/{DD}/{eventId}.jpg
 3) 이미지 업로드
3. MQTT 연결 해제
'''


'''
Lambda를 통해 AI 카메라 디바이스가 MQTT로 보낸 이벤트를 받아서 메타데이터를 DynamoDB에 기록하고, S3에 Presigned URL을 생성하여 디바이스가 이미지를 업로드할 수 있도록 합니다.
또한, 이미지가 업로드되면 생성형 AI를 통해 이미지 분석을 하여 현재 상태를 message로 기록합니다.

[DynamoDB 스키마 (`events` 테이블)]
| 필드 | 타입 | 예시 | 비고 |
|---|---|---|---|
| `pk` | string (PK) | `SITE#OCTANK-1#CAM#3F-07` | 사이트·카메라 단위 파티션 키 |
| `sk` | string (SK) | `EVT#2025-07-27T08:15:23.412Z#ID#542991` | 시간 정렬 + `eventId` 포함 |
| `eventId` | string | `542991` | 멱등 키(고유 이벤트 ID) |
| `ts` | string (ISO8601) | `2025-07-27T08:15:23.412Z` | UTC 권장 |
| `eventType` | string | `DANGER_ZONE_INTRUSION` | 이벤트 유형 |
| `severity` | string | `HIGH` | 심각도 |
| `message` | string | `작업자 A가 탱크구역 A 폴리곤 내부 진입` | 자유 텍스트 설명(이미지분석) |
| `roiId` | string | `TANK-AREA-A` | 지오펜스/구역 ID |
| `model` | map | `{ "name":"yolov8n", "ver":"1.3.2", "conf":0.82 }` | 추론 모델 메타 |
| `image.required` | bool | `true` | 이미지 필요 여부 |
| `image.s3Key` | string | `OCTANK-1/3F-07/2025/07/27/542991.jpg` | 사전 결정 S3 키 |
| `thumbKey` | string | `.../542991_thumb.jpg` | 썸네일 S3 키(선택) |
| `status` | string | `PENDING_IMAGE → IMAGE_ATTACHED` | 상태 머신 |
| `createdAt` | string | `2025-07-27T08:15:23.900Z` | 레코드 생성 시각 |


- 입력: IoT Rule 이벤트(JSON)
- 동작:
    - PutItem(조건: attribute_not_exists(eventId))로 멱등 선기록
    - S3 Presigned PUT URL 생성
    - 디바이스 응답 토픽에 URL 발송
- 출력: { eventId, s3Key, url, expireSec }

- 트리거: S3:ObjectCreated:*
- 동작:
    - UpdateItem으로 status=IMAGE_ATTACHED, imageS3Key 갱신
    - 실패 시: DLQ(SQS) 로 이동, 재처리 잡 제공
'''

import json
import os
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, TypedDict

# Configuration
BUCKET_NAME = os.getenv("BUCKET_NAME", "SAMPLE")
DUMMY_DB_PATH = Path("data_source/dynamodb_anomaly_data/dummy_safety_events_2025.json")
PRESIGNED_EXP_SEC = 300

DEVICE_IDS = [
    "1F-01", "1F-03", "1F-06", "2F-02", "2F-04", "2F-07",
    "3F-03", "3F-05", "3F-07", "3F-11", "4F-07", "4F-12",
    "5F-01", "5F-08",
]
EVENT_TYPES = [
    "DANGER_ZONE_INTRUSION", "VEHICLE_ENTERED",
    "UNAUTHORIZED_ACCESS", "FIRE_ALERT", "NO_ENTRY_VIOLATION",
]
SEVERITIES = ["LOW", "MEDIUM", "HIGH"]
MODEL_INFO = {"name": "yolov8n", "ver": "1.3.2", "conf": 0.82}
ROI_IDS = ["TANK-AREA-A", "TANK-AREA-B", "FUEL-STORAGE-AREA",
           "SECURITY-CHECK-POINT", "LOADING-BAY"]
WORKERS = ["A", "B", "C", "D"]
VEHICLES = ["Truck", "Forklift", "Van"]
LOCATIONS = ["Tank Area A", "Fuel Storage Area", "Loading Bay",
             "Security Check Point"]

# Schema
class ImageInfo(TypedDict):
    required: bool
    s3Key: str

class EventItem(TypedDict, total=False):
    pk: str
    sk: str
    eventId: str
    siteId: str
    deviceId: str
    ts: str
    eventType: str
    severity: str
    message: str
    roiId: str
    model: Dict[str, Any]
    image: ImageInfo
    status: str
    createdAt: str

# Fucntions
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def random_timestamp(start: datetime, end: datetime) -> str:
    delta = end - start
    rand_sec = random.randint(0, int(delta.total_seconds()))
    return (start + timedelta(seconds=rand_sec)).strftime("%Y-%m-%dT%H:%M:%SZ")

def ensure_db_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def load_db(path: Path) -> List[EventItem]:
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8") as fp:
            data = json.load(fp)
            return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []

def save_db(path: Path, items: List[EventItem]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(items, fp, ensure_ascii=False, indent=2)

def build_s3_key(event_type: str) -> str:
    return f"data_source/anomaly_images_s3/{event_type}.png"

def generate_message(event_type: str, roi_id: str, worker: str,
                     vehicle: str, location: str) -> str:
    templates = {
        "DANGER_ZONE_INTRUSION": f"{worker} entered the {roi_id} danger‑zone polygon.",
        "VEHICLE_ENTERED":      f"Vehicle {vehicle} entered the {roi_id} area.",
        "UNAUTHORIZED_ACCESS":  f"Unauthorized access detected by {worker}.",
        "FIRE_ALERT":           f"Fire alert triggered at {location}.",
        "NO_ENTRY_VIOLATION":   f"No‑entry violation by {worker}.",
    }
    return templates[event_type]

def append_to_dummy_db(item: EventItem) -> None:
    ensure_db_path(DUMMY_DB_PATH)
    items = load_db(DUMMY_DB_PATH)

    if any(d.get("eventId") == item["eventId"] for d in items):  # dedupe
        return

    items.append(item)
    save_db(DUMMY_DB_PATH, items)

# Lambda handler
def lambda_handler(event: Dict[str, Any] | str,
                   context: Any = None) -> Dict[str, Any]:

    if isinstance(event, str):
        event = json.loads(event)

    site_id    = event["siteId"]
    device_id  = event["deviceId"]
    event_id   = event.get("eventId", str(uuid.uuid4()))
    ts         = event.get("ts", now_iso())
    event_type = event["eventType"]

    s3_key     = build_s3_key(event_type)

    # Build DynamoDB item (metadata) 
    item: EventItem = {
        "pk": f"SITE#{site_id}#CAM#{device_id}",
        "sk": f"EVT#{ts}#ID#{event_id}",
        "eventId":   event_id,
        "siteId":    site_id,
        "deviceId":  device_id,
        "ts":        ts,
        "eventType": event_type,
        "severity":  event["severity"],
        "message":   event["message"],
        "roiId":     event["roiId"],
        "model":     event["model"],
        "image": {
            "required": bool(event.get("imageRequired", True)),
            "s3Key": s3_key,
        },
        "status":    "ACTIVATE",
        "createdAt": now_iso(),
    }

    # Actual implementation would call boto3 DynamoDB here
    append_to_dummy_db(item)

    # Generate (mock) presigned URL
    presigned_url = (
        "https://{bucket}.s3.amazonaws.com/example-image.jpg"
        "?AWSAccessKeyId=ACCESS_KEY&Expires=1753600379"
        "&Signature=eb84dde12739d5e86aedea70a143ddde685d7300e76029e49909ce51faeebd70"
    ).format(bucket=BUCKET_NAME)

    return {
        "statusCode": 200,
        "content": {
            "eventId":   event_id,
            "s3Bucket":  BUCKET_NAME,
            "s3Key":     s3_key,
            "url":       presigned_url,
            "item":      item,
            "headers":   {"Content-Type": "image/jpeg"},
            "expireSec": PRESIGNED_EXP_SEC,
        },
    }

# Synthetic event generator for local testing 
def generate_event_data() -> Dict[str, Any]:
    start = datetime(2025, 7, 20, tzinfo=timezone.utc)
    end   = datetime(2025, 7, 27, 23, 59, 59, tzinfo=timezone.utc)

    event_type = random.choice(EVENT_TYPES)
    roi_id     = random.choice(ROI_IDS)
    worker     = random.choice(WORKERS)
    vehicle    = random.choice(VEHICLES)
    location   = random.choice(LOCATIONS)

    event_payload = {
        "siteId":     "OCTANK-1",
        "deviceId":   random.choice(DEVICE_IDS),
        "eventId":    str(random.randint(500_000, 600_000)),
        "ts":         random_timestamp(start, end),
        "eventType":  event_type,
        "severity":   random.choice(SEVERITIES),
        "message":    generate_message(event_type, roi_id, worker, vehicle, location),
        "roiId":      roi_id,
        "model":      MODEL_INFO,
        "imageRequired": True,
    }

    # Immediately invoke handler for easy manual testing
    return lambda_handler(event_payload, None)
