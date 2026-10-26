import * as vscode from 'vscode';

/**
 * Interpreter selected in the Microsoft Python extension, if installed.
 * Never activates or installs anything that the user did not install.
 */
export async function pythonExtensionInterpreter(onChange: () => void, subs: vscode.Disposable[]): Promise<string | undefined> {
  const ext = vscode.extensions.getExtension('ms-python.python');
  if (!ext) {
    return undefined;
  }
  try {
    const { PythonExtension } = await import('@vscode/python-extension');
    const api = await PythonExtension.api();
    subs.push(api.environments.onDidChangeActiveEnvironmentPath(() => onChange()));
    const envPath = api.environments.getActiveEnvironmentPath(vscode.workspace.workspaceFolders?.[0]?.uri);
    const resolved = await api.environments.resolveEnvironment(envPath);
    return resolved?.executable.uri?.fsPath ?? envPath.path;
  } catch {
    return undefined;
  }
}
