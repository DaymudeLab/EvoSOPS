#!/bin/bash
##Run this only if code changes
cargo build -r
./target/release/swarm_aggregation_ga -b sep -e ga -g 1600 -p 600 --gran 10 -m 0.005 '-k(13,9)' -s566372 -s31728 --path ./output/Agg_genome.log
