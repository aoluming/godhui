import cv2

videoCapture = cv2.VideoCapture('./0617.avi')
destFps = 10
fps = videoCapture.get(cv2.CAP_PROP_FPS)
if fps != destFps:
    frameSize = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # 这里的VideoWriter_fourcc需要多测试，如果编码器不对则会提示报错，根据报错信息修改编码器即可
    videoWriter = cv2.VideoWriter('./', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), destFps, frameSize)

    i = 0;
    while True:
        success, frame = videoCapture.read()
        if success:
            i += 1
            print('转换到第%d帧' % i)
            videoWriter.write(frame)
        else:
            print('帧率转换结束')
            break