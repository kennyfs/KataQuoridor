/*
 * walleffects.cpp
 *
 *   katago quoridorwalleffects -input rows.bin -output effects.bin
 *
 * Computes the Quoridor I/O v4 raw wall effects (QuoridorNN::fillCanonicalWallEffectU8) for rows given by the canonical
 * (side-to-move-relative) planes the training data stores, with the same function the training-data writer uses. This is
 * what python/quoridor_convert_tdata_v3_to_v4.py calls to add wallEffectNCHW to v3 rows (docs/QuoridorIOv4.md).
 *
 * rows.bin: back to back 130-byte records: [0] the side to move's pawn cell, [1] the opponent's pawn cell (both
 * canonical row * 9 + column), [2..65] placed horizontal wall anchors, [66..129] placed vertical wall anchors (both
 * indexed canonical row * 8 + column, 0 / 1). effects.bin: back to back 324-byte records, 4 x 9 x 9, as in the writer.
 * The canonical orientation is the board seen with the side to move as Black (heading for row 0), so the position is
 * rebuilt as a board with Black to move.
 */

#include "../core/global.h"
#include "../game/board.h"
#include "../neuralnet/quoridornn.h"
#include "../command/commandline.h"
#include "../main.h"

#include <cstdio>

using namespace std;

int MainCmds::quoridorwalleffects(const vector<string>& args) {
  Board::initHash();

  string inputFile;
  string outputFile;
  try {
    KataGoCommandLine cmd("Compute Quoridor I/O v4 wall effects for rows of canonical planes (training data conversion).");
    TCLAP::ValueArg<string> inputArg("", "input", "Input rows file", true, string(), "FILE");
    TCLAP::ValueArg<string> outputArg("", "output", "Output file", true, string(), "FILE");
    cmd.add(inputArg);
    cmd.add(outputArg);
    cmd.parseArgs(args);
    inputFile = inputArg.getValue();
    outputFile = outputArg.getValue();
  }
  catch(TCLAP::ArgException& e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  FILE* in = fopen(inputFile.c_str(), "rb");
  FILE* out = fopen(outputFile.c_str(), "wb");
  if(in == NULL || out == NULL) {
    cerr << "Could not open " << (in == NULL ? inputFile : outputFile) << endl;
    return 1;
  }
  const int RECORD_IN = 2 + 64 + 64;
  const int RECORD_OUT = QuoridorNN::NUM_WALL_EFFECT_CHANNELS * 81;
  uint8_t rec[RECORD_IN];
  uint8_t res[RECORD_OUT];
  int64_t numRows = 0;
  while(true) {
    size_t got = fread(rec, 1, RECORD_IN, in);
    if(got == 0)
      break;
    if(got != (size_t)RECORD_IN) {
      cerr << "Truncated input record" << endl;
      return 1;
    }
    Board board;
    // Walls first, alternating the placer so that neither runs out of fences (up to 20 walls).
    int placed = 0;
    for(int r = 0; r < 8; r++) {
      for(int c = 0; c < 8; c++) {
        if(rec[2 + r * 8 + c]) {
          board.playMoveAssumeLegal(Location::hWallLoc(c, r, board.x_size), placed % 2 == 0 ? P_BLACK : P_WHITE);
          placed++;
        }
        if(rec[66 + r * 8 + c]) {
          board.playMoveAssumeLegal(Location::vWallLoc(c, r, board.x_size), placed % 2 == 0 ? P_BLACK : P_WHITE);
          placed++;
        }
      }
    }
    if(placed > 20) {
      cerr << "More than 20 walls in a row" << endl;
      return 1;
    }
    board.playMoveAssumeLegal(Location::pawnLoc(rec[0] % 9, rec[0] / 9, board.x_size), P_BLACK);
    board.playMoveAssumeLegal(Location::pawnLoc(rec[1] % 9, rec[1] / 9, board.x_size), P_WHITE);
    QuoridorNN::fillCanonicalWallEffectU8(board, P_BLACK, res);
    fwrite(res, 1, RECORD_OUT, out);
    numRows++;
  }
  fclose(in);
  fclose(out);
  cout << "Wrote wall effects for " << numRows << " rows to " << outputFile << endl;
  return 0;
}
