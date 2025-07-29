import os
import sys
import json
import streamlit as st
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

DB_DIR = "data_source/dynamodb_anomaly_data"
PDF_DIR = "data_source/s3_work_instruction_pdf"
DATA_FILE = f"{DB_DIR}/dummy_safety_events_2025.json" # 이벤트 JSON 데이터 파일 경로 (예시)

def save_json_event(data, filepath:str=DATA_FILE):
    # 기존 데이터 로드 또는 빈 리스트
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    # 데이터 추가 및 저장
    existing_data.append(data)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)

    return filepath


# 페이지 기본 설정
st.set_page_config(page_title="SafeGuard AI Dashboard", layout="wide")

st.title("📹 SafeGuard AI Dashboard")
st.write("카메라별 실시간 상태, 이벤트 발생이력 및 AI 분석 리포트를 확인합니다.")

# 1. 이벤트 데이터 불러오기 (로컬 JSON 파일)
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        events = json.load(f)
else:
    st.error(f"데이터 파일을 찾을 수 없습니다: {DATA_FILE}")
    st.stop()

# JSON 데이터가 리스트 형식의 이벤트 모음이라고 가정
if not isinstance(events, list):
    st.error("이벤트 데이터 형식이 올바르지 않습니다. 리스트 형태여야 합니다.")
    st.stop()

# 문자열 타임스탬프를 datetime으로 변환하여 정렬 (최신 이벤트 먼저)
for evt in events:
    try:
        evt_time = datetime.fromisoformat(evt["ts"].replace("Z", "+00:00"))
    except Exception:
        evt_time = datetime.min
    evt["__parsed_time"] = evt_time


# 이벤트를 시간 기준으로 내림차순 정렬 (최신순)
events.sort(key=lambda x: x["__parsed_time"], reverse=True)


# 2. 사이드바 - 필터 위젯
camera_list = ["(전체)"] + sorted({ evt["deviceId"] for evt in events })
severity_list = ["(전체)"] + sorted({ evt["severity"] for evt in events })

# 이벤트 감지 (예시생성)
st.sidebar.header("이벤트 생성")
with st.spinner("분석 중입니다... 잠시만 기다려 주세요."):
    if st.sidebar.button("이벤트 감지하기"):
        # 이벤트 감지 함수 호출
        import lambda_function_event
        response = lambda_function_event.generate_event_data()
        st.success(f"이벤트가 감지되었습니다.")

        print(response)
        print(response["content"])
        print(response["content"]["item"])
        # EventBridge에 이벤트 전송후 RAG Pipeline 실행
        sys.path.append(os.path.abspath("ecs-rag-pipeline"))
        import workflow
        workflow.run_rag_pipeline(response["content"]["item"])
        st.write(f"분석완료하였습니다. ")    
        try:
            st.rerun()
        except AttributeError:
            st.experimental_rerun()

# 파일업로드
st.sidebar.header("임베딩")
uploaded_file = st.sidebar.file_uploader("작업지시서를 업로드해주세요.", type=["pdf"], key="pdf_uploader")  

# 업로드된 파일 처리  
import lambda_function_embedding 
if uploaded_file is not None:  
    file_name = uploaded_file.name 
    save_path = os.path.join(PDF_DIR, file_name)    
    with open(save_path, "wb") as f:  
        f.write(uploaded_file.read()) 

    event = {
        "file_name": file_name
    }     
     # 로딩 상태 표시  
    with st.spinner("파일을 저장하는 중입니다... 잠시만 기다려 주세요."):
        response = lambda_function_embedding.lambda_handler(event, None)     
    st.session_state.pop("pdf_uploader", None)
    st.sidebar.info("임베딩 완료! 새 파일을 선택하세요.")
    st.write(f"파일이 성공적으로 임베딩되었습니다: {response}")  
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()
        
# 사이드바에 필터 선택박스 추가
st.sidebar.header("필터")
camera_filter = st.sidebar.selectbox("카메라 선택", camera_list)
severity_filter = st.sidebar.selectbox("심각도 선택", severity_list)

# 선택한 필터를 적용하여 이벤트 목록 필터링
filtered_events = []

for evt in events:
    if camera_filter != "(전체)" and evt["deviceId"] != camera_filter:
        continue
    if severity_filter != "(전체)" and evt["severity"] != severity_filter:
        continue
    filtered_events.append(evt)

# 3. 대시보드 내용 - 카메라별 섹션 출력
if not filtered_events:
    st.write("선택된 조건에 해당하는 이벤트가 없습니다.")
else:
    # 필터된 이벤트들을 카메라별로 그룹화
    cameras = {}
    for evt in filtered_events:
        cam = evt["deviceId"]
        cameras.setdefault(cam, []).append(evt)
    # 각 카메라별로 섹션 생성
    for cam, cam_events in cameras.items():
        # 카메라 섹션 헤더
        latest_evt = cam_events[0] # 시간순 정렬된 상태에서 첫 번째가 최신 이벤트
        latest_status = latest_evt["status"]
        latest_time = latest_evt["ts"]
        latest_sev = latest_evt["severity"]
        # 상태에 따라 텍스트에 색상 마크다운 적용
        status_color = {"ACTIVATE": "🟢", "DEACTIVATE": "🟠"}
        status_text = status_color.get(latest_status, latest_status)
        # 심각도에 따라 텍스트 적용
        sev_color = {"HIGH": "**HIGH**", "MEDIUM": "**MEDIUM**", "LOW": "**LOW**"}
        sev_text = sev_color.get(latest_sev, latest_sev)
        st.markdown(f"### 카메라[{status_text}] **{cam}** - 최근 이벤트: {latest_time} ({sev_text})")
        # 카메라 이벤트 이력 나열
        for evt in cam_events:
            t = evt["ts"].replace("T", " ").replace("Z", "")
            event_type = evt.get("eventType", "")
            sev = evt.get("severity", "")
            msg = evt.get("message", "")

            # 한 줄 요약 정보
            st.write(f"- **[{t}] {event_type}** (심각도: {sev}) - {msg}")
            # 세부 정보 (AI 분석 리포트 등) expander로 표시
            with st.expander("자세히 보기", expanded=False):
                # AI 분석 리포트 존재 여부 확인
                advisor = evt.get("ragAdvisor")
                if advisor:
                    st.write(advisor)
                    # st.write("**AI 권고 조치 계획:**")
                    # # Action Plan 단계별 표시
                    # for step in advisor.get("actionPlan", []):
                    #     st.write(f"    {step['priority']}. {step['step']}")
                    # # 설명 추가
                    # if advisor.get("explanation"):
                    #     st.write(f"**사유 설명:** {advisor['explanation']}")
                    # # 인용된 안전 수칙 등 (citations) 표시
                    # citations = advisor.get("citations")
                    # if citations:
                    #     st.write(f"*참고 지침:* {', '.join(citations)}")
                else:
                    st.write("AI 분석 정보가 없습니다.")
                # 관련 이미지 표시
                img_path = evt.get("imageS3Key")
                if img_path and os.path.exists(img_path):
                    st.image(img_path, caption=f"이벤트 ID {evt.get('eventId')} 관련 이미지")




