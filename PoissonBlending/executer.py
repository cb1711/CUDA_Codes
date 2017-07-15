import os
os.system("nvcc blend.cu")
os.system("img2Mat.exe")
print "1"
os.system("a.exe")
print "2"
os.system("mat2img.exe")
print "3"

