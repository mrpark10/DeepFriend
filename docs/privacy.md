# DeepFriend 개인정보처리방침

시행일: 2026년 10월 26일 · 운영자: 박준호 (sk92977852@gmail.com)

## 1. 수집하는 정보
DeepFriend 개발자는 **어떠한 개인정보도 수집하거나 서버로 전송하지 않아요.** DeepFriend에는 사용 통계·광고·분석 기능이 없어요.

## 2. 내 컴퓨터에 저장되는 정보
- `deepfriend.watch()`를 사용한 학습의 설정값, loss·정확도 기록, 학습 코드 사본, 데이터 샘플 몇 개(작게 줄인 값)가 프로젝트의 `.deepfriend/` 폴더에 저장돼요.
- 이 폴더에는 자동으로 `.gitignore`가 만들어져 Git에 실수로 올라가지 않아요.
- 언제든 폴더를 지우거나 설정 `deepfriend.autoRecord`를 꺼서 저장을 멈출 수 있어요.

## 3. AI 기능을 켠 경우 (선택)
- 설정 `deepfriend.ai.provider`에서 AI를 고르면, 사용자가 허락한 경우에만 **에러 메시지와 관련 코드 최대 10줄**, 또는 **학습 진단 문장**이 사용자가 고른 AI 제공자(VS Code 내장 AI/GitHub Copilot, Anthropic, 또는 사용자가 지정한 OpenAI 호환 주소)로 전송돼요.
- 보내기 전에 보낼 내용을 그대로 보여주고 "이번만 허용 / 항상 허용 / 사용 안 함"을 고르게 해요.
- 전송된 내용은 해당 AI 제공자의 개인정보처리방침에 따라 처리돼요. DeepFriend 개발자는 그 내용을 받거나 볼 수 없어요.
- API 키는 VS Code 보안 저장소(SecretStorage)에만 저장돼요.

## 4. 오류 신고
"오류 신고하기"를 누르면 버전 정보가 채워진 GitHub Issue 작성 화면이 열려요. 내용은 사용자가 직접 확인하고 제출하며, 자동으로 전송되지 않아요. 사용자 이름이 들어간 경로는 `~`로 가려요.

## 5. 문의
sk92977852@gmail.com 또는 https://github.com/mrpark10/DeepFriend/issues
