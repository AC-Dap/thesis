#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>      // for std::iota
#include <random>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>      // for memcpy
#include <cstdlib>      // for exit
#include <tuple>
#include <unordered_set>

// Constants for header sizes
const size_t GLOBAL_HEADER_SIZE = 24;
const size_t PACKET_HEADER_SIZE = 16;

struct SampledPacket {
    uint32_t src_addr;
    uint32_t dest_addr;
};

// This function samples packets from a pcap file.
// It returns a vector where each element is a vector of raw packet bytes.
std::vector<SampledPacket> sample_pcap(const std::string &filename, size_t n_packets,
                                       size_t n_samples = 1000000) {
    // Generate a pool of packet indices [0, n_packets - 1]
    std::vector<size_t> pool(n_packets);
    std::iota(pool.begin(), pool.end(), 0);

    // Set up a random number generator
    std::random_device rd;
    std::mt19937 rng(rd());

    // Shuffle and pick the first n_samples, then sort them.
    std::shuffle(pool.begin(), pool.end(), rng);
    std::vector<size_t> indices(pool.begin(), pool.begin() + n_samples);
    std::sort(indices.begin(), indices.end());

    // Open the file and memory map it.
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1) {
        perror("open");
        exit(EXIT_FAILURE);
    }
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        perror("fstat");
        close(fd);
        exit(EXIT_FAILURE);
    }
    size_t file_size = sb.st_size;
    char* data = static_cast<char*>(mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0));
    if (data == MAP_FAILED) {
        perror("mmap");
        close(fd);
        exit(EXIT_FAILURE);
    }
    close(fd);  // file descriptor no longer needed after mapping

    std::vector<SampledPacket> sampled_packets;
    sampled_packets.reserve(n_samples);

    size_t offset = GLOBAL_HEADER_SIZE; // Skip the global header.
    size_t curr_packet = 0;
    size_t curr_sample = 0;

    // Iterate through the file until we've sampled the desired number of packets
    while (curr_sample < n_samples && offset < file_size) {
        // Ensure there is enough room for a packet header.
        if (offset + PACKET_HEADER_SIZE > file_size)
            break;

        // Read the packet length (assume 1 byte at offset + 8)
        int packet_len = static_cast<unsigned char>(data[offset + 8]);

        // Ensure we have enough bytes for the packet data.
        if (offset + PACKET_HEADER_SIZE + packet_len > file_size)
            break;

        // If the current packet index matches one of our sampled indices,
        // copy the raw packet data (ignoring IP parsing)
        if (curr_packet == indices[curr_sample]) {
            std::vector<uint8_t> packet(packet_len);
            memcpy(packet.data(), data + offset + PACKET_HEADER_SIZE, packet_len);

	        SampledPacket sp;
            memcpy(&sp.src_addr, &packet[12], sizeof(sp.src_addr));
            memcpy(&sp.dest_addr, &packet[16], sizeof(sp.dest_addr));

            sampled_packets.push_back(sp);
            curr_sample++;
        }

        curr_packet++;
        offset += PACKET_HEADER_SIZE + packet_len;
    }

    munmap(data, file_size);
    return sampled_packets;
}

int main() {
    std::vector<std::tuple<std::string, size_t>> files = {
      {"20020814-090000-0-anon", 22497005},
      {"20020814-090500-0-anon", 22420502},
      {"20020814-091000-0-anon", 22504496},
      {"20020814-091500-0-anon", 22610876},
      {"20020814-092000-0-anon", 23102771},
      {"20020814-092500-0-anon", 22787973},
      {"20020814-093000-0-anon", 22590033},
      {"20020814-093500-0-anon", 22810815},
      {"20020814-094000-0-anon", 24894670},
      {"20020814-094500-0-anon", 23100165},
      {"20020814-095000-0-anon", 22891427},
      {"20020814-095500-0-anon", 22375813},
    };

    for (auto [filename, n_packets] : files) {
      	auto pcap_file = "data/CAIDA/" + filename + ".pcap";
        auto packets = sample_pcap(pcap_file, n_packets);

        // Write sampled packets to file
        std::ofstream f("data/processed/CAIDA/orig-" + filename + ".csv");
    	if (!f.is_open()) {
        	std::cerr << "Failed to open output file" << std::endl;
        	return 1;
    	}
    	f << "src,dest" << std::endl;

        for (const auto &packet : packets) {
            f << packet.src_addr << "," << packet.dest_addr << std::endl;
        }
        f.close();

        std::cout << "Processed " << filename << std::endl;
    }

    return 0;
}

