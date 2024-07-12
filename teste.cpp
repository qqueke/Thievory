#include <iostream>
#include <limits>

void func() { return; }

int main() {
  // Unsigned int
  std::cout << "Range of unsigned int: "
            << std::numeric_limits<unsigned int>::min() << " to "
            << std::numeric_limits<unsigned int>::max() << std::endl;

  // Unsigned long
  std::cout << "Range of unsigned long: "
            << std::numeric_limits<unsigned long>::min() << " to "
            << std::numeric_limits<unsigned long>::max() << std::endl;

  // Unsigned long long
  std::cout << "Range of unsigned long long: "
            << std::numeric_limits<unsigned long long>::min() << " to "
            << std::numeric_limits<unsigned long long>::max() << std::endl;

  unsigned int a = 8;
  std::cout << "Result of 4 divded by sizeof(unsigned int) = "
            << a / sizeof(unsigned int) << std::endl;

  std::cout << "Prefix ++ result:  = " << ++a << std::endl;

  func();

  unsigned long long *array;

  std::cout << "Size of our data type: " << sizeof(*array) << std::endl;

  return 0;
}
