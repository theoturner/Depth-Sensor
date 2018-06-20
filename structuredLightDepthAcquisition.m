function structuredLightDepthAcquisition

    close all;
    
    % Set to 'synthetic,' 'real' or 'own'
    dataType = input('Specify data type (synthetic, real or own): ', 's');

    if strcmp(dataType, 'synthetic')

        % Change to what you'd like
        frames = loadFrames('data/cube_T1', '00', 0, 39, 2, 'png');
        
        % Increase this to reduce false positives but increase false
        % negatives
        bgNoiseLevel = 0.1;

    elseif strcmp(dataType, 'real')

        % Change to what you'd like
        frames = loadFrames('data/real_ball_tea', 'IMG_95', 01, 40, 2, 'jpg');
        
        % Increase this to reduce false positives but increase false
        % negatives
        bgNoiseLevel = 0.3;

    elseif strcmp(dataType, 'own')

        % Change to what you'd like
        frames = loadFrames('ownData', 'IMG_28', 41, 80, 2, 'jpg');
        
        % Increase this to reduce false positives but increase false
        % negatives
        bgNoiseLevel = 0.8;

    end

    uvCodes = getUVCodes(frames, bgNoiseLevel);
    
    % Change this to what you'd like
    calibrationMatrix = loadFromJSON('real.matrices');
    
    depthMap = getDepth(uvCodes, calibrationMatrix);

    % Plot point cloud
    [xs, ys, zs] = deal(depthMap(:, :, 1), depthMap(:, :, 2), depthMap(:, :, 3));
    
    %{
    % Uncomment and enter desired matrix to compare scatter plots
    calibrationMatrix2 = loadFromJSON('data/ownSynthetic.matrices');
    depthMap2 = getDepth(uvCodes, calibrationMatrix2);
    [x2s, y2s, z2s] = deal(depthMap2(:, :, 1), depthMap2(:, :, 2), depthMap2(:, :, 3));
    %}
    
    scatter3(xs(:), ys(:), zs(:), 'r.');
    
    %{
    % Uncomment to compare scatter plots
    hold on;
    scatter3(x2s(:), y2s(:), z2s(:), 'b.');
    hold off;
    %}

    runProjections(dataType);

end



% =========================================================================
% RECONSTRUCTION ==========================================================
% =========================================================================

% Decode the light patterns using image differencing to get (u, v) codes at
% each pixel. Codes are stored as a height x width x 2 matrix, with the two
% layers of the third dimension corresponding to u and v decimal values.
% Unreliable values are marked using a code of -1 to avoid having the same
% decimal code as any reliable pixel (since decimals cannot be negative).
function uvCodes = getUVCodes(frames, bgNoiseLevel)

    vertical = size(frames, 1);
    horizontal = size(frames, 2);
    numberOfFrames = size(frames, 4);
    uvCodes = zeros(vertical, horizontal, 2);

    % Convert frames to grayscale and then get differences between pairs of
    % resulting frames
    grayFrames = zeros(vertical, horizontal, numberOfFrames);
    tempFrames = zeros(vertical, horizontal, 0.5 * numberOfFrames);
    for frameCount = 1 : numberOfFrames

        grayFrames(:, :, frameCount) = rgb2gray(frames(:, :, :, frameCount));

        % If even, get difference - we're working with each pair in sequence,
        % so have half the number of values
        if ~bitget(frameCount, 1)

            tempFrames(:, :, 0.5 * frameCount) = grayFrames(:, :, frameCount - 1) - grayFrames(:, :, frameCount);

        end

    end

    numberOfPairs = size(tempFrames, 3);
    
    % Do this here to save it being done every time the decimal is
    % generated (all our decimals are the same length)
    powersOfTwo = generatePowers(0.5 * numberOfPairs);

    for verticalCount = 1 : vertical

        for horizontalCount = 1 : horizontal

            % Compute differences
            pixelDifference = reshape(tempFrames(verticalCount, horizontalCount, :), 1, numberOfPairs);
            differenceSum = abs(sum(pixelDifference));

            % Mark pixels whose gray code can not be determined reliably
            % with -1 to avoid having the same decimal code as any reliable
            % pixel (since decimals cannot be negative).
            if differenceSum < bgNoiseLevel

                uvCodes(verticalCount, horizontalCount, :) = -1;

            % If reliable, compute code
            else

                % Make binary, 1 if difference >= 0
                for count = 1 : numberOfPairs
                    
                    if pixelDifference(count) >= 0
                        
                        pixelDifference(count) = 1;
                        
                    else
                        
                        pixelDifference(count) = 0;
                        
                    end
                    
                end

                % Convert binary to decimal
                uBinary = pixelDifference(1 : (0.5 * numberOfPairs));
                uDecimal = binaryToDecimal(uBinary, powersOfTwo);
                vBinary = pixelDifference(((0.5 * numberOfPairs) + 1) : numberOfPairs);
                vDecimal = binaryToDecimal(vBinary, powersOfTwo);

                uvCodes(verticalCount, horizontalCount, :) = [uDecimal, vDecimal];

            end

        end

    end

end


% The following two functions are for converting a binary array to a
% decimal value. We do this so we can store a single decimal value for each
% of u and v, which is more elegant than storing binary. Decimal values can
% be easily converted back. Note that there are a pair of built-in
% functions for doing this in MATLAB:
% binaryString = num2str(binaryArray);
% decimal = bin2dec(binaryString);
% However, this pairing is surprisingly inefficient. As we are dealing with
% large image matrices, I have manually implemented this function to
% improve speed. The reason my implementation is split over two functions
% is that the size of the decimal value is the same for every pixel so we
% only need to generate the appropriate number of powers of two once
% instead of on every binary to decimal conversion.
function decimal = binaryToDecimal(binaryArray, powersOfTwo)
    
    binaryPowers = binaryArray .* powersOfTwo;
    decimal = sum(binaryPowers);
        
end
function powersOfTwo = generatePowers(sizeOfDecimal)

    powersOfTwo = zeros(1, sizeOfDecimal);
    for count = 1 : sizeOfDecimal
        
        powersOfTwo(count) = 2^(count - 1);
        
    end

end


% Using the provided calibration matrices, for each pixel remaining in the
% image, determine the unique depth that minimises the distance to the ray
% passing through the projector that is consistent with the (u, v) code.
% Afterward, compute the depth-map for the provided datasets. This function
% is based on an algorithm for inferring 3D world points from Simon
% Prince's book 'Computer Vision:  Models, Learning, and Inference' (2012).
% Given N calibrated cameras in known positions, viewing the same
% three-dimensional point w and knowing the corresponding projections in
% a set of images x_i, the algorithm establish the position of the point in
% the world.
function depthMap = getDepth(uvCodes, calibrationMatrix)

    vertical = size(uvCodes, 1);
    horizontal = size(uvCodes, 2);
    depthMap = zeros(vertical, horizontal, 3);

    cameraIntrinsic  = calibrationMatrix(1).intrinsic;
    projectorIntrinsic = calibrationMatrix(2).intrinsic;
    cameraExtrinsic  = calibrationMatrix(1).extrinsic;
    projectorExtrinsic = calibrationMatrix(2).extrinsic;
    
    cameraRotationCells = num2cell(cameraExtrinsic(1 : 9));
    [w111, w211, w311, w121, w221, w231, w131, w321, w331] = cameraRotationCells{:};
    
    projectorRotationCells = num2cell(projectorExtrinsic(1 : 9));
    [w112, w212, w312, w122, w222, w232, w132, w322, w332] = projectorRotationCells{:};
    
    cameraTranslationCells = num2cell(cameraExtrinsic(10 : end));
    [tx1, ty1, tz1] = cameraTranslationCells{:};
    
    projectorTranslationCells = num2cell(projectorExtrinsic(10 : end));
    [tx2, ty2, tz2] = projectorTranslationCells{:};
    
    for verticalCount = 1 : vertical
        
        for horizontalCount = 1 : horizontal
            
            thisUVCode = uvCodes(verticalCount, horizontalCount, :);
            
            % If it's a reliable pixel, compute depth, otherwise depth defaults to 0 
            if thisUVCode ~= -1

                % Only two image points, camera and projector
                
                % Convert points to normalized camera coordinates
                xy1 = cameraIntrinsic \ [horizontalCount; verticalCount; 1];
                xy2 = projectorIntrinsic \ [thisUVCode(2); thisUVCode(1); 1];

                % Compute linear constraints
                a11 = [w311 * xy1(1) - w111, w231 * xy1(1) - w121, w331 * xy1(1) - w131];
                a21 = [w311 * xy1(2) - w211, w231 * xy1(2) - w221, w331 * xy1(2) - w321];
                b1 = [tx1 - tz1 * xy1(1); ty1 - tz1 * xy1(2)];
                a12 = [w312 * xy2(1) - w112, w232 * xy2(1) - w122, w332 * xy2(1) - w132];
                a22 = [w312 * xy2(2) - w212, w232 * xy2(2) - w222, w332 * xy2(2) - w322];
                b2 = [tx2 - tz2 * xy2(1); ty2 - tz2 * xy2(2)];

                % Stack linear constraints
                A = [a11; a21; a12; a22];
                b = [b1; b2];
                
                % Least squares solution for parameters
                w = (A' * A) \ A' * b;

                % Go to camera perspective (pre-multiply by the extrinsic
                % matrix)
                depthMap(verticalCount, horizontalCount, :) = cameraExtrinsic * [w; 1];
                
            end
            
        end
        
    end

end


% Simple function for decoding JSON files
function structure = loadFromJSON(filePath)

    file = fopen(filePath); 
    data = fread(file, inf); 
    dataString = char(data'); 
    fclose(file); 
    structure = jsondecode(dataString);

end


% The following two functions are for loading in frames. They are taken
% from load_sequence_color but have alterations to make the matrices
% produced easier to deal with.
function output = loadFrames(path, prefix, first, last, digits, suffix)

    % Get the padded frame number
    number = padNumber(first, digits);

    % Check for slash at the end of the path
    if(path(end) == '/')
        
        slash = '';
        
    else
        
        slash = '/';
        
    end

    % Create the filename
    filename = strcat(path, slash, prefix, number, '.', suffix);

    % Load image and convert it to gray level, resize to make matrix
    % manageable
    current = imresize(im2double(imread(filename)), 0.25);

    % Create output matrix
    output = zeros(size(current, 1), size(current, 2), 3, last - first + 1);
    output(:, :, :, 1) = current;

    for count = 2 : last - first + 1

        % Get the padded frame number
        number = padNumber(first + count - 1, digits);

        % Create the filename
        filename = strcat(path, slash, prefix, number, '.', suffix);

        % Load image and convert it to gray level, resize to make matrix
        % manageable
        current = imresize(im2double(imread(filename)), 0.25);

        % Update output matrix
        output(:, :, :, count) = current;


    end

end
function output = padNumber(number, digits)

    % Convert to string
    output = num2str(number);

    % Get length of string
    length = size(output, 2);

    % Add necessary zeros
    for count = length + 1 : digits

        output = strcat('0', output);

    end

end


% =========================================================================
% END OF RECONSTRUCTION ===================================================
% =========================================================================



% =========================================================================
% CALIBRATION =============================================================
% =========================================================================

% This function runs the reprojection function in the appropriate way for
% the data type and saves the outputs.
function runProjections(dataType)

    if strcmp(dataType, 'synthetic')
        
        % This variable stores information about the checkerboard position and
        % size. The first item is the x-positon, the second is the y-position
        % and the third is the width of the checkerboard. The variable is
        % the same in the following if statements as well.
        checkerboardData = [378, 277, 270];
        projectorResolution = [1024, 768];
       
        for imageCount = [0, 2, 4]
            
            projectedFrame = imread(strcat('data/calibration/000', num2str(imageCount), '.png'));
            outputFrame = imread(strcat('data/calibration/000', num2str(imageCount + 1), '.png'));
            
            projectedFrameOfReference = imref2d(size(projectedFrame), [1, projectorResolution(1)], [1, projectorResolution(2)]);
            
            correctedOutputFrame = project(projectedFrame, outputFrame, checkerboardData, projectedFrameOfReference);
            imwrite(correctedOutputFrame, strcat('data/calibration/projected_000', num2str(imageCount + 1), '.png'));
            
        end
        
    elseif strcmp(dataType, 'real')
        
        checkerboardData = [518, 120, 299];
        projectorResolution = [1024, 768];
        
        for imageCount = 1 : 9
            
            % Both the projected and output frames are the same in the case
            % of non-synthetic data.
            [projectedFrame, outputFrame] = deal(imread(strcat('data/real_calibration/IMG_932', num2str(imageCount), '.jpg')));
            
            projectedFrameOfReference = imref2d(size(projectedFrame), [1, projectorResolution(1)], [1, projectorResolution(2)]);
            
            correctedOutputFrame = project(projectedFrame, outputFrame, checkerboardData, projectedFrameOfReference);
            imwrite(correctedOutputFrame, strcat('data/real_calibration/p-projected_IMG_932', num2str(imageCount), '.jpg'));
            
        end
        
    elseif strcmp(dataType, 'own')
        
        checkerboardData = [690, 270, 540];
        projectorResolution = [1920, 1080];
        
        for imageCount = 1 : 6
            
            % Both the projected and output frames are the same in the case
            % of non-synthetic data, as above.
            [projectedFrame, outputFrame] = deal(imread(strcat('ownCalibration/IMG_0', num2str(imageCount), '.jpg')));
            
            projectedFrameOfReference = imref2d(size(projectedFrame), [1, projectorResolution(1)], [1, projectorResolution(2)]);
            
            correctedOutputFrame = project(projectedFrame, outputFrame, checkerboardData, projectedFrameOfReference);
            imwrite(correctedOutputFrame, strcat('ownCalibration/projected_IMG_0', num2str(imageCount), '.jpg'));
            
        end
        
    else
        
        fprintf('Error: dataType has been manually changed.\n')
        
    end

end


% Reproject the checkerboards after getting the corners from user input.
function correctedOutputFrame = project(projectedFrame, outputFrame, checkerboardData, projectedFrameOfReference)

    % Get corner co-ordinates from user input
    corners = getUserInput(projectedFrame);
    
    % Find homography and project - taken from Gabriel Brostow's Machine
    % Vision course, see below.
    pts1Cart =  [corners(1, :); corners(2, :)];
    pts2Cart =  [checkerboardData(1), checkerboardData(1) + checkerboardData(3), checkerboardData(1) + checkerboardData(3), checkerboardData(1); ...
                 checkerboardData(2), checkerboardData(2), checkerboardData(2) + checkerboardData(3), checkerboardData(2) + checkerboardData(3)];
    homography = calculateBestHomography(pts1Cart, pts2Cart)';
    
    % Express homography as a projection
    transform = projective2d(homography);
    
    % Apply transform
    correctedOutputFrame = imwarp(outputFrame, transform, 'OutputView', projectedFrameOfReference);
    
end


% Get user to select corners of the checkerboard. Read using ginput and
% stored in a 2D array.
function corners = getUserInput(projectedFrame)

    corners = zeros(4, 2);
    fprintf('Click on the corners of the checkerboard.\n');
    figure;
    imshow(projectedFrame);
    hold on;
    for count = 1 : 4
        
        [horizontalPosition, verticalPosition] = ginput(1);
        plot(horizontalPosition, verticalPosition, 'g+');
        corners(count, :) = [horizontalPosition, verticalPosition];
        
    end
    close;
    corners = corners';

end


% The following two functions are used to calculate the best homography
% between two sets of points. They are taken directly from my solutions to
% homework assignments from Gabriel Brostow's Machine Vision course at UCL.
function homography = calculateBestHomography(pts1Cart, pts2Cart)

    pts1Cart = [pts1Cart; ones(1, size(pts1Cart, 2))];
    pts2Cart = [pts2Cart; ones(1, size(pts2Cart, 2))];
    A = zeros(2 * size(pts1Cart, 2), 9);
    
    for count = 1 : size(pts1Cart, 2)

        ucount = pts1Cart(1, count);
        vcount = pts1Cart(2, count);
        xcount = pts2Cart(1, count);
        ycount = pts2Cart(2, count);
        A(2 * count - 1, :) = [0, 0, 0, -ucount, -vcount, -1, ycount * ucount, ycount * vcount, ycount];
        A(2 * count, :) = [ucount, vcount, 1, 0, 0, 0, -xcount * ucount, -xcount * vcount, -xcount];

    end

    homography = solveAXEqualsZero(A); 
    homography = reshape(homography, [3, 3])';

end
function x = solveAXEqualsZero(A)

    [~, ~, V] = svd(A);
    x = V(:, end);

end

% =========================================================================
% END OF CALIBRATION ======================================================
% =========================================================================
