# 배포 안내 (게시자 계정 만들기)

DeepFriend는 `v*` 태그를 올리면 GitHub Actions가 VS Code 마켓플레이스·Open VSX·PyPI에 자동으로 올려요. 처음 한 번만 아래 계정과 토큰이 필요해요.

## 1. VS Code Marketplace (필수)
1. https://marketplace.visualstudio.com/manage 에 Microsoft 계정으로 로그인
2. **Create publisher** → ID: `mrpark10`, 이름: `박준호` (또는 원하는 표시 이름)
3. https://dev.azure.com 에서 같은 계정으로 조직을 하나 만들고(이름 자유) 오른쪽 위 사용자 아이콘 → **Personal access tokens** → **New Token**
   - Organization: **All accessible organizations**
   - Scopes: **Custom defined** → **Marketplace** → **Manage** 체크
   - 만료일은 원하는 만큼(최대 1년)
4. 만들어진 토큰을 복사해 두세요 (`VSCE_PAT`)

## 2. Open VSX (Cursor·Windsurf 사용자용, 권장)
1. https://open-vsx.org 에 GitHub 계정으로 로그인
2. 프로필 → **Settings** → **Publisher Agreement** 동의(Eclipse 계정 연결)
3. **Access Tokens** → 새 토큰 생성 (`OVSX_PAT`)
4. 처음 한 번 네임스페이스 생성: `npx ovsx create-namespace mrpark10 -p <OVSX_PAT>`

## 3. PyPI (Jupyter·외부 터미널 사용자용, 권장)
1. https://pypi.org 계정 만들기(2단계 인증 필요)
2. **Your projects → Publishing → Add a new pending publisher**
   - PyPI project name: `deepfriend` · Owner: `mrpark10` · Repository: `DeepFriend` · Workflow: `release.yml` · Environment: `pypi`
3. GitHub 저장소 → Settings → Environments → **New environment** `pypi`

## 4. 토큰을 GitHub에 넣기
```bash
gh secret set VSCE_PAT --repo mrpark10/DeepFriend
gh secret set OVSX_PAT --repo mrpark10/DeepFriend
```

## 5. 출시
```bash
git tag v0.1.0 && git push origin v0.1.0
```
