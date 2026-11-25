package com.boardcheck.app

import android.os.Handler
import android.os.Looper
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString.Companion.toByteString
import org.json.JSONObject
import java.util.concurrent.TimeUnit

/**
 * Link to the laptop bridge (server/bridge.py), which relays to the website.
 *
 *   Wi-Fi / hotspot: the URL comes from the QR code on the website, ws://<laptop-ip>:8765/phone
 *   USB cable:       ws://127.0.0.1:8765/phone after `adb reverse tcp:8765 tcp:8765`
 *
 * Phone -> website: status, frame (binary JPEG), detections, advice_token, advice_done.
 * Website -> phone: {"type":"command","cmd":"start|stop|capture|thresholds", ...}.
 * Only results and camera frames leave the phone; all AI runs here.
 */
class BridgeClient(
    private val onState: (Boolean) -> Unit,
    private val onCommand: (JSONObject) -> Unit,
) {
    private val http = OkHttpClient.Builder().pingInterval(10, TimeUnit.SECONDS).connectTimeout(2, TimeUnit.SECONDS).build()
    private val main = Handler(Looper.getMainLooper())
    private var ws: WebSocket? = null
    private var url: String? = null
    @Volatile var connected = false
        private set

    fun connect(target: String) {
        url = target
        ws?.cancel()
        open()
    }

    fun disconnect() { url = null; ws?.close(1000, "bye"); ws = null; setConnected(false) }

    private fun open() {
        val u = url ?: return
        ws = http.newWebSocket(Request.Builder().url(u).build(), object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) = setConnected(true)
            override fun onMessage(webSocket: WebSocket, text: String) {
                runCatching { JSONObject(text) }.getOrNull()?.takeIf { it.optString("type") == "command" }
                    ?.let { main.post { onCommand(it) } }
            }
            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) = retry(webSocket)
            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) = retry(webSocket)
        })
    }

    private fun retry(socket: WebSocket) {
        if (socket !== ws) return
        setConnected(false)
        if (url != null) main.postDelayed({ if (url != null && !connected) open() }, 1000)
    }

    private fun setConnected(v: Boolean) { connected = v; main.post { onState(v) } }

    val target: String? get() = url

    fun send(json: JSONObject) { if (connected) ws?.send(json.toString()) }
    fun sendFrame(jpeg: ByteArray) { if (connected) ws?.send(jpeg.toByteString()) }
}
