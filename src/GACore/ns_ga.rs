use super::Genome;
use std::collections::HashMap;

pub struct  NSGA {
    max_gen: u16,
    elitist_cnt: u16,
    population: Vec<Genome>,
    mut_rate: f64,
    granularity: u16,
    genome_cache: HashMap<[u16; 6], f64>,
    perform_cross: bool,
    sizes: Vec<(u16,u16)>,
    trial_seeds: Vec<u64>
}