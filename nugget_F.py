from model_sim import *
import matplotlib.pyplot as plt
import numpy as np
F_X_vectorized = np.vectorize(lib.F_X)
pmixture_C_vectorized = np.vectorize(lib.pmixture_C)




x = np.linspace(0,400,51)
# x = np.linspace(20000,200000,101)
# x = np.linspace(36.82, 36.8205, 500)
# for xi in x:
    # print(xi, lib.my_F(xi,1,2)) # breaks after 93800

fig, ax = plt.subplots()

tao = [1,2,4,8,16,32]
for tao_i in tao:
    print(tao_i)
    F_X = F_X_vectorized(x,1,2,tao_i)
    plt.plot(x, F_X, marker='o', linestyle='-', label=r'$F_{X}(x)$, $\tau=$'+str(tao_i))

F_X_star = pmixture_C_vectorized(x,1,2)
plt.plot(x, F_X_star, marker='o', linestyle='-',label=r'$F_{X*}(x)$')


# F_X = F_X_vectorized(x,1,2,10)
# plt.plot(x,F_X,'bo-',label='part1+part2')

# F_X_star = pmixture_C_vectorized(x,1,2)
# plt.plot(x,F_X_star,'ro-', label=r'1 - $F_{X*}(x)$')

# basically same result as pmixture_C
# my_F_vectorized = np.vectorize(lib.my_F)
# integral_my_F = 1 - my_F_vectorized(x,1,2)
# plt.plot(x, integral_my_F, 'bo', label=r'my 1 - $F_{X*}(x)$')

# F_X_part_1_vectorized = np.vectorize(lib.F_X_part_1)
# integral_nugget_part_1 = F_X_part_1_vectorized(x,1,2,1)
# plt.plot(x, integral_nugget_part_1, color='orange', marker='o', linestyle='-', label='part1')

# F_X_part_2_vectorized = np.vectorize(lib.F_X_part_2)
# integral_nugget_part_2 = F_X_part_2_vectorized(x,1,2,1)
# plt.plot(x,integral_nugget_part_2, 'go-', label='part2')
legend = ax.legend(loc='lower right', fontsize='x-large')

plt.show()

# fig, ax = plt.subplots()

# diff = integral_nugget - integral
# plt.plot(x, diff, 'bo-')
# plt.xlim(0,35)
# plt.ylim(-0.05,0.05)
# plt.show()