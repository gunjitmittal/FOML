%% Import data from text file.
% READ_DRIVPOINTS Import data from a text file as column vectors.
%
%   [FILENAME,SUBJECT,IMGNUM,LABEL,ANG,XF,YF,WF,HF,XRE,YRE,XLE,YLE,XN,YN,XRM,YRM,XLM,YLM]
%   = READ_DRIVPOINTS(FILENAME) Reads data from text file FILENAME for the
%   default selection.
%
%   [FILENAME,SUBJECT,IMGNUM,LABEL,ANG,XF,YF,WF,HF,XRE,YRE,XLE,YLE,XN,YN,XRM,YRM,XLM,YLM]
%   = READ_DRIVPOINTS(FILENAME, STARTROW, ENDROW) Reads data from rows STARTROW
%   through ENDROW of text file FILENAME.
%
% Example:
%   [fileName,subject,imgNum,label,ang,xF,yF,wF,hF,xRE,yRE,xLE,yLE,xN,yN,xRM,yRM,xLM,yLM] = read_drivPoints('drivPoints.txt',1, 606);
%
% You can also use: 
%    T = readtable('drivPoints.txt');
%
% Kt Diaz - 05/05/5016
%%------------------------------------------- 

function [fileName,subject,imgNum,label,ang,xF,yF,wF,hF,xRE,yRE,xLE,yLE,xN,yN,xRM,yRM,xLM,yLM] = read_drivPoints(filename, startRow, endRow)

%% Initialize variables.
delimiter = ',';
if nargin<=1
    startRow = 1;
    endRow = inf;
else
    endRow = endRow +1;
end
startRow = startRow+1;

%% Read columns of data as strings:
% For more information, see the TEXTSCAN documentation.
formatSpec = '%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%[^\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');

%% Read columns of data according to format string.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
dataArray = textscan(fileID, formatSpec, endRow(1)-startRow(1)+1, 'Delimiter', delimiter, 'MultipleDelimsAsOne', true, 'HeaderLines', startRow(1)-1, 'ReturnOnError', false);
for block=2:length(startRow)
    frewind(fileID);
    dataArrayBlock = textscan(fileID, formatSpec, endRow(block)-startRow(block)+1, 'Delimiter', delimiter, 'MultipleDelimsAsOne', true, 'HeaderLines', startRow(block)-1, 'ReturnOnError', false);
    for col=1:length(dataArray)
        dataArray{col} = [dataArray{col};dataArrayBlock{col}];
    end
end

%% Close the text file.
fclose(fileID);

%% Convert the contents of columns containing numeric strings to numbers.
% Replace non-numeric strings with NaN.
raw = repmat({''},length(dataArray{1}),length(dataArray)-1);
for col=1:length(dataArray)-1
    raw(1:length(dataArray{col}),col) = dataArray{col};
end
numericData = NaN(size(dataArray{1},1),size(dataArray,2));

for col=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    % Converts strings in the input cell array to numbers. Replaced non-numeric
    % strings with NaN.
    rawData = dataArray{col};
    for row=1:size(rawData, 1);
        % Create a regular expression to detect and remove non-numeric prefixes and
        % suffixes.
        regexstr = '(?<prefix>.*?)(?<numbers>([-]*(\d+[\,]*)+[\.]{0,1}\d*[eEdD]{0,1}[-+]*\d*[i]{0,1})|([-]*(\d+[\,]*)*[\.]{1,1}\d+[eEdD]{0,1}[-+]*\d*[i]{0,1}))(?<suffix>.*)';
        try
            result = regexp(rawData{row}, regexstr, 'names');
            numbers = result.numbers;
            
            % Detected commas in non-thousand locations.
            invalidThousandsSeparator = false;
            if any(numbers==',');
                thousandsRegExp = '^\d+?(\,\d{3})*\.{0,1}\d*$';
                if isempty(regexp(thousandsRegExp, ',', 'once'));
                    numbers = NaN;
                    invalidThousandsSeparator = true;
                end
            end
            % Convert numeric strings to numbers.
            if ~invalidThousandsSeparator;
                numbers = textscan(strrep(numbers, ',', ''), '%f');
                numericData(row, col) = numbers{1};
                raw{row, col} = numbers{1};
            end
        catch me
        end
    end
end


%% Split data into numeric and cell columns.
rawNumericColumns = raw(:, [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]);
rawCellColumns = raw(:, 1);


%% Replace non-numeric cells with NaN
R = cellfun(@(x) ~isnumeric(x) && ~islogical(x),rawNumericColumns); % Find non-numeric cells
rawNumericColumns(R) = {NaN}; % Replace non-numeric cells

%% Allocate imported array to column variable names
fileName = rawCellColumns(:, 1);
subject = cell2mat(rawNumericColumns(:, 1));
imgNum = cell2mat(rawNumericColumns(:, 2));
label = cell2mat(rawNumericColumns(:, 3));
ang = cell2mat(rawNumericColumns(:, 4));
xF = cell2mat(rawNumericColumns(:, 5));
yF = cell2mat(rawNumericColumns(:, 6));
wF = cell2mat(rawNumericColumns(:, 7));
hF = cell2mat(rawNumericColumns(:, 8));
xRE = cell2mat(rawNumericColumns(:, 9));
yRE = cell2mat(rawNumericColumns(:, 10));
xLE = cell2mat(rawNumericColumns(:, 11));
yLE = cell2mat(rawNumericColumns(:, 12));
xN = cell2mat(rawNumericColumns(:, 13));
yN = cell2mat(rawNumericColumns(:, 14));
xRM = cell2mat(rawNumericColumns(:, 15));
yRM = cell2mat(rawNumericColumns(:, 16));
xLM = cell2mat(rawNumericColumns(:, 17));
yLM = cell2mat(rawNumericColumns(:, 18));


