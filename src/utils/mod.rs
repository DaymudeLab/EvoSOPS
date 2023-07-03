use std::fmt::Write;

pub fn join(a: &[u64]) -> String {
    a.iter().fold(String::new(),|mut s,&n| {write!(s,"_{}",n).ok(); s})
}
