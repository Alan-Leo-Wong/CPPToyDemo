// /*
//  * @Author: Lei Wang 602289550@qq.com
//  * @Date: 2023-03-15 17:37:38
//  * @LastEditors: Lei Wang 602289550@qq.com
//  * @LastEditTime: 2023-03-15 21:41:41
//  * @FilePath: \readFile\main.cpp
//  * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
//  * 进行设置:
//  https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
//  */
// #include <fstream>
// #include <iostream>
// #include <sstream>

// int main() {
//   std::string filePath = "E:\\VSProjects\\3DThinShell\\model\\123.txt";

//   std::ifstream file;
//   file.open(filePath, std::ios::in);

//   char buf[2048] = {0};

//   // while (file.getline(buf, sizeof(buf))) {
//   //   std::stringstream line(buf); // 将line改为字符串
//   //   std::string word;
//   //   line >> word;
//   //   // printf("word length = %d\n", word.length());
//   //   std::cout << word << std::endl;
//   //   double x, y, z;
//   //   line >> x >> y >> z;
//   //   std::cout << x << " " << y << " " << z << std::endl;
//   //   // break;
//   // }
//   std::ofstream s(filePath, std::ios::out);
//   if (!s.is_open()) {
//     fprintf(stderr, "IOError: writeOBJ() could not open %s\n",
//             filePath.c_str());
//   } else {
//     printf("yes\n");
//     s << 1;
//   }
//   return 0;
// }