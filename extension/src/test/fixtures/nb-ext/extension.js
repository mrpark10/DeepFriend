// Test-only helper extension: a notebook type whose cell produces a Python-style error output.
const vscode = require('vscode');

exports.activate = function (context) {
  context.subscriptions.push(
    vscode.workspace.registerNotebookSerializer('df-test-nb', {
      deserializeNotebook: () => new vscode.NotebookData([]),
      serializeNotebook: () => new Uint8Array(),
    }),
  );
  const ctrl = vscode.notebooks.createNotebookController('df-test-ctrl', 'df-test-nb', 'DF test');
  context.subscriptions.push(ctrl);
  context.subscriptions.push(
    vscode.commands.registerCommand('dfTest.errorNotebook', async (name, message, stack) => {
      const nb = await vscode.workspace.openNotebookDocument('df-test-nb', new vscode.NotebookData([new vscode.NotebookCellData(vscode.NotebookCellKind.Code, 'model(x)', 'python')]));
      ctrl.updateNotebookAffinity(nb, vscode.NotebookControllerAffinity.Preferred);
      await vscode.window.showNotebookDocument(nb);
      await vscode.commands.executeCommand('notebook.selectKernel', { id: 'df-test-ctrl', extension: 'deepfriend-test.df-test-notebooks' });
      const exec = ctrl.createNotebookCellExecution(nb.cellAt(0));
      exec.start(Date.now());
      const e = new Error(message);
      e.name = name;
      e.stack = stack;
      await exec.replaceOutput([new vscode.NotebookCellOutput([vscode.NotebookCellOutputItem.error(e)])]);
      exec.end(false, Date.now());
      return true;
    }),
  );
};
