#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>

/** Ensures the bundled Iris dataset is present at the path used by CI (CTORCH_IRIS_CSV). */
TEST(SmokeIris, BundledCsvExists) {
    namespace fs = std::filesystem;
    const fs::path root = fs::path(__FILE__).parent_path().parent_path();
    const fs::path csv = root / "experiments/classification/Iris.csv";
    ASSERT_TRUE(fs::exists(csv)) << csv.string();
    std::ifstream in(csv);
    ASSERT_TRUE(in.good());
}
