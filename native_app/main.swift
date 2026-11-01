import Cocoa
import WebKit

// Neural Edit Pro: a real macOS app window around the editor (no browser). It starts the editor's two servers if they are not running, then shows the editor.
let project = "/Users/gaurisusheela/Downloads/my-ai-photo-app"
let appURL = URL(string: "http://localhost:3210")!

func portUp(_ port: Int) -> Bool {
    let p = Process(); p.executableURL = URL(fileURLWithPath: "/usr/sbin/lsof")
    p.arguments = ["-nP", "-iTCP:\(port)", "-sTCP:LISTEN"]
    let pipe = Pipe(); p.standardOutput = pipe; p.standardError = Pipe()
    try? p.run(); p.waitUntilExit()
    return p.terminationStatus == 0
}

func startServers() {
    let log = NSHomeDirectory() + "/Library/Logs/NeuralEditPro"
    try? FileManager.default.createDirectory(atPath: log, withIntermediateDirectories: true)
    let script = """
    export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
    cd "\(project)" || exit 1
    up() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }
    if ! up 8000; then nohup ./venv/bin/uvicorn server:app --port 8000 >> "\(log)/backend.log" 2>&1 & fi
    if ! up 3210; then
      [ -d pro-editor-frontend/.next-prod ] || (cd pro-editor-frontend && NEP_PROD=1 npm run build >> "\(log)/build.log" 2>&1)
      (cd pro-editor-frontend && NEP_PROD=1 nohup npm run start -- -p 3210 >> "\(log)/frontend.log" 2>&1 &)
    fi
    """
    let p = Process(); p.executableURL = URL(fileURLWithPath: "/bin/bash"); p.arguments = ["-c", script]
    p.standardOutput = Pipe(); p.standardError = Pipe()
    try? p.run()
}

final class AppDelegate: NSObject, NSApplicationDelegate, WKNavigationDelegate, WKUIDelegate, WKDownloadDelegate, NSWindowDelegate {
    var window: NSWindow!
    var web: WKWebView!
    var tries = 0

    func applicationDidFinishLaunching(_ n: Notification) {
        let cfg = WKWebViewConfiguration()
        cfg.preferences.setValue(true, forKey: "developerExtrasEnabled")
        web = WKWebView(frame: .zero, configuration: cfg)
        web.navigationDelegate = self; web.uiDelegate = self
        web.setValue(false, forKey: "drawsBackground")
        let w = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1500, height: 950), styleMask: [.titled, .closable, .miniaturizable, .resizable], backing: .buffered, defer: false)
        w.title = "Neural Edit Pro"; w.backgroundColor = NSColor(red: 0.07, green: 0.07, blue: 0.075, alpha: 1)
        w.contentView = web; w.setFrameAutosaveName("NeuralEditProMain"); w.delegate = self
        if !w.setFrameUsingName("NeuralEditProMain") { w.center() }
        w.makeKeyAndOrderFront(nil); window = w
        buildMenu()
        NSApp.activate(ignoringOtherApps: true)
        web.loadHTMLString(splash("Starting the editor…"), baseURL: nil)
        DispatchQueue.global().async {
            if !(portUp(8000) && portUp(3210)) { startServers() }
            for _ in 0..<150 { if portUp(8000) && portUp(3210) { break }; Thread.sleep(forTimeInterval: 1) }
            DispatchQueue.main.async {
                if portUp(8000) && portUp(3210) { self.web.load(URLRequest(url: appURL)) }
                else { self.web.loadHTMLString(self.splash("The editor could not start. See ~/Library/Logs/NeuralEditPro"), baseURL: nil) }
            }
        }
    }

    func splash(_ msg: String) -> String {
        "<body style='background:#111113;color:#a1a1aa;font:16px -apple-system;display:flex;align-items:center;justify-content:center;height:100vh;margin:0'><div style='text-align:center'><div style='font-size:42px'>⚡</div><div style='margin-top:12px'>\(msg)</div></div></body>"
    }

    func buildMenu() {
        let main = NSMenu(); NSApp.mainMenu = main
        let appItem = NSMenuItem(); main.addItem(appItem); let app = NSMenu(); appItem.submenu = app
        app.addItem(withTitle: "About Neural Edit Pro", action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)), keyEquivalent: "")
        app.addItem(.separator())
        app.addItem(withTitle: "Hide Neural Edit Pro", action: #selector(NSApplication.hide(_:)), keyEquivalent: "h")
        app.addItem(withTitle: "Quit Neural Edit Pro", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        let editItem = NSMenuItem(); main.addItem(editItem); let edit = NSMenu(title: "Edit"); editItem.submenu = edit
        edit.addItem(withTitle: "Undo", action: Selector(("undo:")), keyEquivalent: "z")
        edit.addItem(withTitle: "Redo", action: Selector(("redo:")), keyEquivalent: "Z")
        edit.addItem(.separator())
        edit.addItem(withTitle: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        edit.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        edit.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        edit.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        let viewItem = NSMenuItem(); main.addItem(viewItem); let view = NSMenu(title: "View"); viewItem.submenu = view
        view.addItem(withTitle: "Reload", action: #selector(reload), keyEquivalent: "r")
        view.addItem(withTitle: "Enter Full Screen", action: #selector(NSWindow.toggleFullScreen(_:)), keyEquivalent: "f").keyEquivalentModifierMask = [.command, .control]
        let winItem = NSMenuItem(); main.addItem(winItem); let win = NSMenu(title: "Window"); winItem.submenu = win; NSApp.windowsMenu = win
        win.addItem(withTitle: "Minimize", action: #selector(NSWindow.performMiniaturize(_:)), keyEquivalent: "m")
        win.addItem(withTitle: "Close", action: #selector(NSWindow.performClose(_:)), keyEquivalent: "w")
    }
    @objc func reload() { web.reload() }

    // quitting the window quits the app; the editor's servers keep running (so the phone can still reach it)
    func applicationShouldTerminateAfterLastWindowClosed(_ s: NSApplication) -> Bool { true }
    func applicationShouldHandleReopen(_ s: NSApplication, hasVisibleWindows: Bool) -> Bool { window.makeKeyAndOrderFront(nil); return true }

    // file picker (Open photos…, LUTs)
    func webView(_ webView: WKWebView, runOpenPanelWith params: WKOpenPanelParameters, initiatedByFrame: WKFrameInfo, completionHandler: @escaping ([URL]?) -> Void) {
        let p = NSOpenPanel(); p.allowsMultipleSelection = params.allowsMultipleSelection; p.canChooseDirectories = params.allowsDirectories
        p.beginSheetModal(for: window) { r in completionHandler(r == .OK ? p.urls : nil) }
    }

    // downloads (Export): save into Downloads, then show it in Finder
    func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction, decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        if action.shouldPerformDownload { decisionHandler(.download); return }
        if let u = action.request.url, let host = u.host, host != "localhost", host != "127.0.0.1", u.scheme?.hasPrefix("http") == true { NSWorkspace.shared.open(u); decisionHandler(.cancel); return }
        decisionHandler(.allow)
    }
    func webView(_ webView: WKWebView, decidePolicyFor resp: WKNavigationResponse, decisionHandler: @escaping (WKNavigationResponsePolicy) -> Void) {
        decisionHandler(resp.canShowMIMEType ? .allow : .download)
    }
    func webView(_ webView: WKWebView, navigationAction: WKNavigationAction, didBecome download: WKDownload) { download.delegate = self }
    func webView(_ webView: WKWebView, navigationResponse: WKNavigationResponse, didBecome download: WKDownload) { download.delegate = self }
    func download(_ download: WKDownload, decideDestinationUsing response: URLResponse, suggestedFilename: String, completionHandler: @escaping (URL?) -> Void) {
        let dir = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask)[0]
        var dest = dir.appendingPathComponent(suggestedFilename); var n = 1
        let base = dest.deletingPathExtension().lastPathComponent, ext = dest.pathExtension
        while FileManager.default.fileExists(atPath: dest.path) { n += 1; dest = dir.appendingPathComponent("\(base) \(n)" + (ext.isEmpty ? "" : ".\(ext)")) }
        lastDownload = dest; completionHandler(dest)
    }
    var lastDownload: URL?
    func downloadDidFinish(_ download: WKDownload) { if let u = lastDownload { NSWorkspace.shared.activateFileViewerSelecting([u]) } }
    func download(_ download: WKDownload, didFailWithError error: Error, resumeData: Data?) {
        let a = NSAlert(); a.messageText = "The export could not be saved"; a.informativeText = error.localizedDescription; a.runModal()
    }

    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        tries += 1
        if tries < 20 { DispatchQueue.main.asyncAfter(deadline: .now() + 2) { self.web.load(URLRequest(url: appURL)) } }
    }
    func webView(_ webView: WKWebView, createWebViewWith cfg: WKWebViewConfiguration, for a: WKNavigationAction, windowFeatures: WKWindowFeatures) -> WKWebView? {
        if let u = a.request.url { webView.load(URLRequest(url: u)) }; return nil
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.regular)
let delegate = AppDelegate()
app.delegate = delegate
app.run()
