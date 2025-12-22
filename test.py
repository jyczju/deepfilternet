import platform
print(platform.machine())      # 应该是 'AMD64'
print(platform.architecture()) # 应该是 ('64bit', 'WindowsPE')