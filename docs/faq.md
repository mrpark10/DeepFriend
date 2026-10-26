# 자주 묻는 질문 (FAQ)

**Q. pip로 deepfriend를 설치해야 하나요?**
아니요. VS Code 터미널에서는 확장이 알아서 연결해 줘서 `import deepfriend`가 바로 돼요. Jupyter 노트북이나 VS Code 밖의 터미널에서 쓸 때만 `pip install deepfriend`가 필요해요. (노트북에서 `No module named 'deepfriend'`가 뜨면 해설 카드의 **설치** 버튼을 누르세요.)

**Q. 크기 표시가 안 나와요.**
1) 파일에 `import torch` 또는 `tensorflow`/`keras` import가 있어야 해요. 2) 모델 클래스 바로 위에 `# df: input (...)` 주석이 있어야 해요. 3) 오른쪽 아래 상태 표시줄이 "DeepFriend: 파이썬 필요"라면 파이썬 3.9 이상을 설치하거나 명령 **DeepFriend: 사용할 파이썬 고르기**를 실행하세요. 4) VS Code 설정 `editor.inlayHints.enabled`가 꺼져 있지 않은지 확인하세요.

**Q. `?`가 나와요.**
생성자 인자 값을 모르거나(`__init__(self, n)`), if 문 갈래마다 크기가 다르거나, 아직 지원하지 않는 연산이면 틀린 숫자 대신 `?`를 보여줘요. `?`에 마우스를 올리면 이유가 나와요. 인자에 기본값을 주면 계산할 수 있는 경우가 많아요.

**Q. 학습 코드가 저장할 때마다 실행되나요?**
아니요. 크기를 확인할 때는 import문과 모델 클래스만 골라서, 실제 계산 없이 크기만 계산하는 가짜 텐서(meta)로 확인해요. 학습 루프나 파일 쓰기, 인터넷 접속은 실행되지 않아요.

**Q. 학습 닥터에 검증 loss가 안 보여요.**
검증 loss는 `with torch.no_grad():` 안에서 계산한 손실로 알아봐요. 검증할 때 `torch.no_grad()`를 쓰면 자동으로 잡혀요. 직접 기록하려면 `deepfriend.log(val_loss=..., val_acc=...)`를 쓰세요.

**Q. 실험 기록은 어디에 저장되나요? 지우고 싶어요.**
프로젝트의 `.deepfriend/runs/` 폴더예요. 사이드바의 실험 목록에서 지우거나 폴더를 직접 지우면 돼요. 자동 기록을 끄려면 설정 `deepfriend.autoRecord`를 끄세요.

**Q. AI 키가 꼭 필요한가요?**
아니요. 모든 핵심 기능은 AI 없이 정해진 규칙과 오프라인 사전으로 동작해요. AI는 문장을 더 쉽게 다듬거나, 사전에 없는 에러를 설명할 때만 써요. GitHub Copilot이 있다면 키 없이 쓸 수 있어요.

**Q. Cursor에서도 쓸 수 있나요?**
네. Open VSX에 올라가 있어서 Cursor·Windsurf 등에서 DeepFriend를 검색해 설치할 수 있어요.

**Q. 윈도우에서 DataLoader 오류가 나요.**
윈도우에서 `num_workers`를 0보다 크게 쓰면 학습 코드를 `if __name__ == "__main__":` 안에 넣어야 해요. 아니면 `num_workers=0`으로 바꾸세요.
