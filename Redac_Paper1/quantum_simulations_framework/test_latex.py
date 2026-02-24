import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'nature'])
plt.plot([1, 2], [1, 2])
try:
    plt.savefig('test_latex.png')
    print("SUCCESS")
except Exception as e:
    print(f"FAILED: {e}")
