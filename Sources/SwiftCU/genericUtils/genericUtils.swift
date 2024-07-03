import cxxCU

extension size_t {
    var asMB: Double {
        Double(self) / (1024 * 1024)
    }
}