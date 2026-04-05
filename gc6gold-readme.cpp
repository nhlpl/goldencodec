#include "golden_codec_v6.h"
#include <iostream>
#include <vector>

int main() {
    std::string text = "The future of compression is folding. " + std::string(10000, 'x');
    std::vector<uint8_t> original(text.begin(), text.end());
    auto compressed = GoldenCodecV6::compress(original.data(), original.size());
    auto decompressed = GoldenCodecV6::decompress(compressed.data(), compressed.size());
    std::cout << "Original: " << original.size() << " bytes\n"
              << "Compressed: " << compressed.size() << " bytes\n"
              << "Ratio: " << (double)original.size() / compressed.size() << "x\n";
    bool ok = (original == decompressed);
    std::cout << "Decompression " << (ok ? "SUCCESS" : "FAILURE") << "\n";
    return 0;
}
