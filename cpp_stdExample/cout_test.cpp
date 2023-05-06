#include <iostream>
using namespace std;

int main() {
  int n = 100000;
  for (int i = 0; i < n; i++) {
    printf("[%d\\%d] ", i + 1, n);
    printf("batch_size = %d", i + 2);
    if (i != n - 1)
      printf("\r");
    else
      printf("\n");
  }
  return 0;
}