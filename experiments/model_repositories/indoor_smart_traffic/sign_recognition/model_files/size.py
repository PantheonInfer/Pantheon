import os
def get_FileSize(filePath):
    fsize = os.path.getsize(filePath)
    fsize = fsize / float(1024*1024)
    return round(fsize, 2)

if __name__ == '__main__':
    total = 0
    for file in os.listdir('.'):
        if 'pth' in file:
            total += get_FileSize(file)
    print(total)