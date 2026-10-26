// Error card: forward button clicks to the extension.
(function () {
  const vscode = acquireVsCodeApi();
  document.addEventListener('click', (e) => {
    const el = e.target.closest('[data-action]');
    if (el) {
      vscode.postMessage({ type: el.getAttribute('data-action') });
    }
  });
})();
