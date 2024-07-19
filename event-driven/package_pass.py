
def package_pass_through(loc):
    file_name = f'loc_{loc.ID}.txt'

    # 使用'with'语句打开文件，确保文件会被正确关闭
    # 'w'模式表示写入，如果文件不存在将会被创建
    with open(file_name, 'w', encoding='utf-8') as file:
        # 遍历列表中的每一项
        for i in range(loc.pack_passby.qsize()):
            # 将每一项写入文件，并添加换行符
            file.write(str(loc.pack_passby.get()[1]) + ', ')