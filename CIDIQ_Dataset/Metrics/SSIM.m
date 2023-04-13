function score = SSIM(IO, IR)
    % Convert the input images to grayscale
    IO_gray = im2gray(IO);
    IR_gray = im2gray(IR);

    % Calculate the SSIM score
    [score, ~] = ssim(IO_gray, IR_gray);
end
