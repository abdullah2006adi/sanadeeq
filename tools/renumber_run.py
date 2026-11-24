import sys, json, cv2
sys.path.insert(0,'/home/claude/fix'); from renumber import renumber
WINL = {2: ((0.30, 0.70, 0.60, 0.90), 'dark'), 3: ((0.37, 0.65, 0.04, 0.27), 'light'),
        4: ((0.35, 0.68, 0.0, 0.26), 'dark'), 5: ((0.39, 0.645, 0.25, 0.52), 'light')}
cfg = {b['id']: b for b in json.load(open('/home/claude/sanadeeq/config.json'))['boards']}
for n in (2,3,4,5):
    rr = cfg[f'board{n}']['rects']; ltr = rr[:5][::-1] + rr[5:][::-1]
    img = renumber(cv2.imread(f'/home/claude/sanadeeq/boards/board{n}.jpg'), ltr, WINL[n], [4,3,2,1,0,9,8,7,6,5], loose=True)
    cv2.imwrite(f'/home/claude/fix/board{n}-rtl.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
sys.path.insert(0,'/home/claude/tall2'); import tall2
for n in (2,3,4,5):
    img = renumber(cv2.imread(f'/home/claude/tall2/in{n}.jpg'), tall2.rects(n), tall2.WIN[n], [i ^ 1 for i in range(10)])
    cv2.imwrite(f'/home/claude/fix/board{n}-tall2.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 86])
print('ok')
