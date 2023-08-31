#!/bin/bash
##Run this only if code changes
## cargo build -r
./target/release/swarm_aggregation_ga -b agg -e ga -g 50 -p 100 --gran 10 -m 0.02 '-k(13,9)' -s566372 --path ./output/Agg_genome.log