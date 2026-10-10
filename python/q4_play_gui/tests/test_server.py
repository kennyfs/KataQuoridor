"""The HTTP layer on the mock engine: one tab at a time, JSON only, status codes, the static page, the download."""
import json
import threading
import unittest
import urllib.error
import urllib.request

from helpers import close_game, mock_game
from q4_play_gui.serve import make_server


class Server(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.game = mock_game()
        cls.httpd = make_server(cls.game, "127.0.0.1", 0)
        cls.base = "http://127.0.0.1:%d" % cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        close_game(cls.game)

    def request(self, method, path, body=None, client="tab-one-1234", ctype="application/json"):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method)
        if client:
            req.add_header("X-Client", client)
        if data is not None:
            req.add_header("Content-Type", ctype)
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, r.headers, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers, e.read()

    def post(self, path, body=None, **kw):
        status, _, raw = self.request("POST", path, body or {}, **kw)
        return status, json.loads(raw)

    def test_state_needs_a_client_id(self):
        status, _, _ = self.request("GET", "/api/state", client=None)
        self.assertEqual(status, 400)

    def test_state_and_the_static_page(self):
        status, _, raw = self.request("GET", "/api/state")
        self.assertEqual(status, 200)
        self.assertIn("position", json.loads(raw))
        status, headers, raw = self.request("GET", "/", client=None)
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["Content-Type"])
        self.assertIn(b"<html", raw.lower())

    def test_one_tab_at_a_time(self):
        status, _, _ = self.request("GET", "/api/state", client="tab-two-5678")
        self.assertEqual(status, 423)
        status, _, _ = self.request("GET", "/api/state", client="tab-one-1234")
        self.assertEqual(status, 200)

    def test_posts_need_json(self):
        status, _, _ = self.request("POST", "/api/undo", {}, ctype="text/plain")
        self.assertEqual(status, 415)

    def test_bad_json_and_unknown_endpoint(self):
        req = urllib.request.Request(self.base + "/api/play", data=b"{nope", method="POST")
        req.add_header("X-Client", "tab-one-1234")
        req.add_header("Content-Type", "application/json")
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req)
        self.assertEqual(cm.exception.code, 400)
        status, body = self.post("/api/nothing")
        self.assertEqual(status, 404)

    def test_play_eliminate_undo_redo_flow(self):
        status, body = self.post("/api/new", {"rules": {"maxPlies": 400, "repetitionDrawCount": 0,
                                                        "initialWalls": [7, 7, 7, 7]}})
        self.assertEqual(status, 200)
        self.assertEqual(self.post("/api/play", {"move": "f2"})[0], 200)
        status, body = self.post("/api/play", {"move": "zz"})
        self.assertEqual(status, 400)
        self.assertEqual(self.post("/api/eliminate", {"seat": 1})[0], 200)
        status, body = self.post("/api/eliminate", {"seat": 1})
        self.assertEqual(status, 409)
        self.assertEqual(self.post("/api/undo")[0], 200)
        self.assertEqual(self.post("/api/redo")[0], 200)
        self.assertEqual(self.post("/api/goto", {"ply": 1})[0], 200)
        status, body = self.post("/api/goto", {"ply": 9})
        self.assertEqual(status, 400)

    def test_seats_and_settings_endpoints(self):
        status, _ = self.post("/api/seats", {"seats": [{"kind": "human"}, {"kind": "bot", "bot": "greedy"},
                                                       {"kind": "net", "visits": 50}, {"kind": "human"}],
                                             "me": 3})
        self.assertEqual(status, 200)
        st = json.loads(self.request("GET", "/api/state")[2])
        self.assertEqual(st["seats"][2]["visits"], 50)
        self.assertEqual(st["me"], 3)
        self.assertEqual(self.post("/api/settings", {"maxVisits": 77})[0], 200)
        self.assertEqual(self.post("/api/settings", {"nonsense": 1})[0], 400)
        self.assertEqual(self.post("/api/analysis", {"on": False})[0], 200)

    def test_load_and_download(self):
        status, _ = self.post("/api/load", {"text": "S f2, W b6, EL N"})
        self.assertEqual(status, 200)
        status, headers, raw = self.request("GET", "/api/sgf")
        self.assertEqual(status, 200)
        self.assertIn("attachment", headers["Content-Disposition"])
        self.assertTrue(raw.decode().startswith("(;FF[4]GM[Q4]"))
        status, _ = self.post("/api/load", {"text": "S f2, S f3"})
        self.assertEqual(status, 400)

    def test_legal_for_a_seat(self):
        status, _, raw = self.request("GET", "/api/legal?seat=2")
        self.assertEqual(status, 200)
        self.assertIn("legal", json.loads(raw))
        status, _, _ = self.request("GET", "/api/legal?seat=x")
        self.assertEqual(status, 400)


if __name__ == "__main__":
    unittest.main()
