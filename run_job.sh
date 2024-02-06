#!/bin/bash
##Run this only if code changes
## cargo build -r
./target/release/swarm_aggregation_ga -b brid -e ga -g 500 -p 51 --gran 10 -m 0.08 '-k(13,9)' -s566372 --path ./output/Brid_genome.log