"""Q4 game record reader and writer (JSON Lines format)."""

import json
from typing import Any, Dict, List, Optional
from q4.reference import Pos, action_to_str, eliminate, play, str_to_action


class Q4Record:
    def __init__(self,
                 rules: Optional[Dict[str, Any]] = None,
                 players: Optional[List[Dict[str, Any]]] = None,
                 result: str = "none",
                 events: Optional[List[Dict[str, Any]]] = None,
                 comments: Optional[List[str]] = None):
        self.rules = rules or {
            "maxPlies": 400,
            "repetitionDrawCount": 0,
            "initialWalls": [7, 7, 7, 7],
        }
        self.players = players or [
            {"name": f"P{i+1}", "type": "bot"} for i in range(4)
        ]
        self.result = result
        self.events = events or []
        self.comments = comments or []

    def to_json_line(self) -> str:
        d: Dict[str, Any] = {
            "rules": self.rules,
            "players": self.players,
            "result": self.result,
            "events": self.events,
        }
        if self.comments:
            d["comments"] = self.comments
        return json.dumps(d, separators=(',', ':'))

    @classmethod
    def from_json_line(cls, line: str) -> 'Q4Record':
        d = json.loads(line)
        return cls(
            rules=d.get("rules"),
            players=d.get("players"),
            result=d.get("result", "none"),
            events=d.get("events", []),
            comments=d.get("comments", []),
        )

    def replay(self) -> Pos:
        max_plies = self.rules.get("maxPlies", 400)
        rep = self.rules.get("repetitionDrawCount", 0)
        walls = self.rules.get("initialWalls", [7, 7, 7, 7])
        pos = Pos(max_plies=max_plies, repetition_draw_count=rep, initial_walls=walls)

        for ev in self.events:
            if "elim" in ev:
                seat_1based = ev["elim"]
                pos = eliminate(pos, seat_1based - 1)
            elif "a" in ev:
                action = str_to_action(ev["a"])
                pos = play(pos, action)
            else:
                raise ValueError(f"Unknown event format: {ev}")

        return pos
