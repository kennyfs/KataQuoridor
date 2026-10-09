#ifndef Q4_STYLETOOL_H_
#define Q4_STYLETOOL_H_

#include <string>
#include <vector>

// q4tool subcommands of the style features (docs/q4/Q4IO.md §9):
//   style [-input FILE]: reads game records (JSON lines, stdin by default); for every record replays it with
//     Q4History and prints {"game": i, "numPositions": P, "perspectives": [seat to move at each position],
//     "features": [P * 76 floats]}, the style features at the start position and after every event.
//   popgames -n N [-seed S] [-maxplies M] [-elimprob P] [-bots a,b,c,d] [-targets t0,t1,t2,t3]: random population-like games (each seat a random q4bot of
//     random / randomPawn / greedy / basher / grudge, random eliminations), as game records (JSON lines).
int runQ4StyleCommand(const std::string& subcmd, const std::vector<std::string>& args);
bool isQ4StyleCommand(const std::string& subcmd);

#endif  // Q4_STYLETOOL_H_
