pub mod base_ga;
pub mod ns_ga;

#[derive(Debug, Copy, Clone)]
struct Genome {
    string: [u16; 6],
    fitness: f64,
}
