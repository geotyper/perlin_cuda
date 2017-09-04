#pragma once

#include <cstdint>
#include <ostream>
#include <sstream>
#include <string>
#include <iomanip>
#include "noise_params.hpp"

struct Stats {
	float tMalloc;
	float tKernel;
	float tMemcpy;
	float tTotal;
};

// Pretty print stats
inline std::ostream& operator<<(std::ostream& o, const Stats& stats) {
	const float tot = stats.tTotal;
	const auto getRatio = [tot] (float part) {
		float ratio = part / tot;
		std::stringstream bars;
		bars << "  (" << std::setw(2) << std::setprecision(2) << std::setfill('0') << int(ratio * 100) << "%) ";
		for (unsigned i = 0; i < 30; ++i) {
			if (ratio < i/30.) break;
			bars << '|';
		}
		return bars.str();
	};
//#define FLAGS std::left << std::setprecision(5) << std::setfill('0') << std::setw(7)
	o << "Time: " << "\n";
	o << "    total:  " << stats.tTotal << " ms";
	  //<< "    malloc: " << FLAGS << stats.tMalloc << " ms " << getRatio(stats.tMalloc) << "\n"
	  //<< "    kernel: " << FLAGS << stats.tKernel << " ms " << getRatio(stats.tKernel) << "\n"
	  //<< "    memcpy: " << FLAGS << stats.tMemcpy << " ms " << getRatio(stats.tMemcpy);
//#undef FLAGS
	return o;
}

class PerlinSeq final {
public:
	Stats calculate(uint8_t *hPixels, NoiseParams params);
};
