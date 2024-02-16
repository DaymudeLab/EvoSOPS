#!/bin/bash
##Run this only if code changes
## cargo build -r
./target/release/swarm_aggregation_ga -b brid -e ga -g 10000 -p 600 --gran 10 -m 0.005 '-k(13,9)' -s566372 --path ./output/Brid_genome.log