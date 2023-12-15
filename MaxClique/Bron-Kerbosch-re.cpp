/*
 * @Author: Alan Wang leiw1006@gmail.com
 * @Date: 2023-12-15 20:42:17
 * @LastEditors: Alan Wang leiw1006@gmail.com
 * @LastEditTime: 2023-12-15 20:42:20
 * @FilePath: \MaxClique\Bron-Kerbosch-re.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
 * 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
template <bool isNormalized>
inline void OffsetModel<isNormalized>::computeMaximalClique(
    const std::unordered_map<int, std::unordered_map<int, bool>>
        &noCompatibleGraph,
    std::vector<int> &R, std::vector<int> &P, std::vector<int> &X,
    std::vector<std::vector<int>> &all_maximal_clique) {
  if (P.empty() && X.empty()) {
    all_maximal_clique.emplace_back(R);
  }
  if (P.empty())
    return;

  auto pivot_iter = P.begin();
  int pivot = *pivot_iter;
  while (pivot == -1 && ++pivot_iter != P.end()) {
    pivot = *pivot_iter;
  }
  if (pivot_iter == P.end()) {
    LOG::qpError("NO VALID PIVOT IN 'computeMaximalClique'!");
    exit(1);
  }
  for (auto &p_1 : P) {
    if (p_1 == -1)
      continue;
    // 若pivot和p1相邻, 则continue
    if (pivot != p_1 && (noCompatibleGraph.at(pivot).find(p_1) ==
                             noCompatibleGraph.at(pivot).end() ||
                         !noCompatibleGraph.at(pivot).at(p_1)))
      continue;

    /*std::cout << "pivot = " << pivot << ", p_1 = " << p_1 << std::endl;
    system("pause");*/

    std::vector<int> next_R = R;
    next_R.emplace_back(p_1);

    const auto &p_1_mp = noCompatibleGraph.at(p_1);
    //                std::cout << "have p_1 = " << p_1 << std::endl;
    std::vector<int> next_P;
    for (const auto &p_2 : P)
      if (p_2 != -1 && p_1 != p_2 &&
          (p_1_mp.find(p_2) == p_1_mp.end() ||
           !p_1_mp.at(p_2))) // 保证p_2和p_1相邻(进一步保证了和R中所有元素相邻)
        next_P.emplace_back(p_2);

    std::vector<int> next_X;
    for (const auto &x : X)
      if (x != -1 && p_1 != x &&
          (p_1_mp.find(x) == p_1_mp.end() ||
           !p_1_mp.at(x))) // 保证x和p_1相邻(进一步保证了和R中所有元素相邻)
        next_X.emplace_back(x);

    computeMaximalClique(noCompatibleGraph, next_R, next_P, next_X,
                         all_maximal_clique);

    X.emplace_back(p_1);
    p_1 = -1; // 三角形下标不会为-1
  }
}