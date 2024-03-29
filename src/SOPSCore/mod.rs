pub mod bridging;

#[derive(Eq, PartialEq, Hash)]
struct Particle {
    x: u16,
    y: u16,
    onland: bool,
}
