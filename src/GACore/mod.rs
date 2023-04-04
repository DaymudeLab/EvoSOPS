pub mod base_ga;
pub mod ns_ga;
// pub mod seg_ga;

#[derive(Debug, Copy, Clone)]
struct Genome {
    string: [u16; 6],
    fitness: f64,
}

#[derive(Debug, Copy, Clone)]
struct SegGenome {
    string: [[u16; 6]; 7],
    fitness: f64,
}