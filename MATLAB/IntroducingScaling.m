function IntroducingScaling(DataSetS, LengthBinMethodS, iHistogramAPPos)

%% Extracting embryos

if ~exist('DataSetS', 'var') || isempty(DataSetS), DataSetS = 'TempVaried'; end
if ~exist('LengthBinMethodS', 'var') || isempty(LengthBinMethodS), LengthBinMethodS = 'DoubleGaussian'; end
if ~exist('iHistogramAPPos', 'var') || isempty(iHistogramAPPos), iHistogramAPPos = 401; end


%P = GlobalProps;
switch DataSetS
    case 'LE&SE'
        ProfileD = load('ScalingData1And23.mat');
        iM = 2;
        
        OrigFigurePathS = 'C:\E\Dropbox\Scaling paper\Figures\Fig1ScalingAffectsAPPatterningWithinFlySpecies';
        OrigFig1_D = load([OrigFigurePathS, filesep, 'Fig1.mat']);
        FullEmLengthR = [OrigFig1_D.ELdata.M(13).Data];
        
    case 'LargeSet'
        ProfileD = load('ScalingDataLargeSetBcd.mat');
        iM = 1;
        
        FullEmLengthR = [ProfileD.RawData.M.Em.EL];
        
        
    case 'TempVaried'
        ProfileD = load('ScalingDataTempVariedBcd.mat');
        iM = 1;
        
        FullEmLengthR = [ProfileD.RawData.M.Em.EL];
        
end
iGene = find(strcmp(ProfileD.RawData.M(iM).Genename, 'Bcd'));
ManualTimeRangeR = [130, 190];
% I'm really not sure if we want to select all of nc14, but let's keep it
% like this for now...
DVFlag = 1;
% DVFlag = 1, dorsalprofile; DVFlag = 2, ventralprofile
SmoothSpan = 51;

PosC = (0:0.001:0.999)';
iPosRangeC = (51:951)';
HistogramAPPos = PosC(iHistogramAPPos);


%%% Select embryos

TotalNumGenes = length(ProfileD.RawData.M(iM).Genename);

FullEmAgeR = [ProfileD.RawData.M(iM).Em.Emage];
EmAgeFilterLR = FullEmAgeR >= ManualTimeRangeR(1) & FullEmAgeR <= ManualTimeRangeR(2);

FullEmFlagAllGeneR = cell2mat({ProfileD.RawData.M(iM).Em.Emflag});
FullEmFlagR = FullEmFlagAllGeneR(iGene:TotalNumGenes:end);
EmFlagFilterLR = logical(FullEmFlagR);

EmFilterLR = EmAgeFilterLR & EmFlagFilterLR;
EmLengthR = FullEmLengthR(EmFilterLR);


%%% Manipulating profiles

NumEms = sum(EmFilterLR);
ProfileAllGenes3M = cell2mat({ProfileD.RawData.M(iM).Em(EmFilterLR).Profile});
RawProfileM = ProfileAllGenes3M(:, iGene:TotalNumGenes:end, DVFlag);
ProfileM = NaN(size(RawProfileM));
for lEm = 1:NumEms
    RawProfileM(:, lEm) = inpaint_nans(RawProfileM(:, lEm));
    ProfileM(:, lEm) = smooth(RawProfileM(:, lEm), SmoothSpan);
end


%%% Normalizing profiles

MeanMax = mean(max(ProfileM(iPosRangeC, :), [], 1));
NormProfileM = ProfileM / MeanMax;
MeanNormProfileC = mean(NormProfileM, 2);
BcdConcThreshold = MeanNormProfileC(iHistogramAPPos);



%% Determining the threshold egg length between long and short embryos and
%%% separating the embryos into length classes

% Note: depending on the method, SEFilterR and LEFilterR are either vectors
% of booleans or vectors indexing certain embryos

switch LengthBinMethodS
    case 'DoubleGaussian'
        assert(strcmp(DataSetS, 'LE&SE') | strcmp(DataSetS, 'TempVaried'));
        
        obj = gmdistribution.fit(FullEmLengthR', 2,...
            'Start', struct('mu', [451; 521], 'Sigma', 20, 'PComponents', [0.5, 0.5]),...
            'SharedCov', true);
        Mean1 = obj.mu(1); SD1 = obj.Sigma; Mean2 = obj.mu(2); SD2 = obj.Sigma;
        Threshold = (SD1*Mean2 + SD2*Mean1) / (SD1 + SD2);
        % Derived from finding where the magnitude of the z-scores will be
        % equal
        LEFilterR = EmLengthR >= Threshold;
        SEFilterR = ~LEFilterR;
    case 'ExtremeDeciles'
        [~, iSortedEmLengthR] = sort(EmLengthR);
        NumEmsInBin = ceil(length(EmLengthR)/10);
        SEFilterR = iSortedEmLengthR(1:NumEmsInBin);
        LEFilterR = iSortedEmLengthR(end-NumEmsInBin+1:end);
    case 'Mean'
        Threshold = mean(FullEmLengthR);
        LEFilterR = EmLengthR >= Threshold;
        SEFilterR = ~LEFilterR;
    case 'Median'
        [~, iSortedEmLengthR] = sort(EmLengthR);
        SEFilterR = iSortedEmLengthR(1:ceil(length(EmLengthR)/2));
        LEFilterR = iSortedEmLengthR(ceil(length(EmLengthR)/2)+1:end);
end



%% Averaging profiles
AbsStepSize = 0.5; % In microns
AbsPosC = 0:AbsStepSize:max(EmLengthR)+AbsStepSize;

RawAbsPosM = repmat(EmLengthR, length(PosC), 1) .* repmat(PosC, 1, length(EmLengthR));
InterpAbsPosM = NaN(length(AbsPosC), length(EmLengthR));
for lEm = 1:length(EmLengthR)
    InterpAbsPosM(:, lEm) = interp1(RawAbsPosM(:, lEm), NormProfileM(:, lEm), AbsPosC);
end
MeanAbsSEProfileC = nanmean(InterpAbsPosM(:, SEFilterR), 2);
SD_AbsSEProfileC = nanstd(InterpAbsPosM(:, SEFilterR), [], 2);
MeanAbsLEProfileC = nanmean(InterpAbsPosM(:, LEFilterR), 2);
SD_AbsLEProfileC = nanstd(InterpAbsPosM(:, LEFilterR), [], 2);

MeanSEProfileC = nanmean(NormProfileM(:, SEFilterR), 2);
SD_SEProfileC = nanstd(NormProfileM(:, SEFilterR), [], 2);
MeanLEProfileC = nanmean(NormProfileM(:, LEFilterR), 2);
SD_LEProfileC = nanstd(NormProfileM(:, LEFilterR), [], 2);



%% Generating histogram at left
HistogramFluorescenceR = NormProfileM(iHistogramAPPos, :);
FluorescenceC = (0:0.001:0.999)';
iLeftHistPosC = (1:25:1000)';
SELeftHistValueC = (hist(HistogramFluorescenceR(SEFilterR), FluorescenceC(iLeftHistPosC)))' / sum(SEFilterR);
LELeftHistValueC = (hist(HistogramFluorescenceR(LEFilterR), FluorescenceC(iLeftHistPosC)))' / sum(LEFilterR);

SELeftGuideLineR = [mean(HistogramFluorescenceR(SEFilterR)) - std(HistogramFluorescenceR(SEFilterR)),...
    mean(HistogramFluorescenceR(SEFilterR)),...
    mean(HistogramFluorescenceR(SEFilterR)) + std(HistogramFluorescenceR(SEFilterR))];
LELeftGuideLineR = [mean(HistogramFluorescenceR(LEFilterR)) - std(HistogramFluorescenceR(LEFilterR)),...
    mean(HistogramFluorescenceR(LEFilterR)),...
    mean(HistogramFluorescenceR(LEFilterR)) + std(HistogramFluorescenceR(LEFilterR))];
SELeftGaussC = CreateGaussianFit(FluorescenceC, iLeftHistPosC, SELeftHistValueC);
LELeftGaussC = CreateGaussianFit(FluorescenceC, iLeftHistPosC, LELeftHistValueC);



%% Generating histogram at bottom
ThresholdPosR = NaN(size(EmLengthR));
iDecreasingC = (151:951)';
for lEm = 1:length(EmLengthR)
    try
        ThresholdPosR(lEm) = interp1(NormProfileM(iDecreasingC, lEm), PosC(iDecreasingC), BcdConcThreshold);
    catch
        SmoothSpan = 81;
        PassL = false;
        while ~PassL
            try
                %fprintf('Trying SmoothSpan at %d for embryo %d.\n', SmoothSpan, lEm);
                NewProfileC = smooth(RawProfileM(:, lEm), SmoothSpan);
                ThresholdPosR(lEm) = interp1(NewProfileC(iDecreasingC)/MeanMax, PosC(iDecreasingC), BcdConcThreshold);
                PassL = true;
            catch
                SmoothSpan = SmoothSpan + 20;
            end
        end
    end
end
iHistPosC = 1:25:1000;
if sum(SEFilterR == 0)
    % This is a kludge to determine if we're working with a boolean array!
    SEBottomHistValueC = (hist(ThresholdPosR(SEFilterR), PosC(iHistPosC)))' / sum(SEFilterR);
    LEBottomHistValueC = (hist(ThresholdPosR(LEFilterR), PosC(iHistPosC)))' / sum(LEFilterR);
else
    SEBottomHistValueC = (hist(ThresholdPosR(SEFilterR), PosC(iHistPosC)))' / length(SEFilterR);
    LEBottomHistValueC = (hist(ThresholdPosR(LEFilterR), PosC(iHistPosC)))' / length(LEFilterR);
end

SEBottomGuideLineR = [mean(ThresholdPosR(SEFilterR)) - std(ThresholdPosR(SEFilterR)),...
    mean(ThresholdPosR(SEFilterR)),...
    mean(ThresholdPosR(SEFilterR)) + std(ThresholdPosR(SEFilterR))];
LEBottomGuideLineR = [mean(ThresholdPosR(LEFilterR)) - std(ThresholdPosR(LEFilterR)),...
    mean(ThresholdPosR(LEFilterR)),...
    mean(ThresholdPosR(LEFilterR)) + std(ThresholdPosR(LEFilterR))];
SEBottomGaussC = CreateGaussianFit(PosC, iHistPosC, SEBottomHistValueC);
LEBottomGaussC = CreateGaussianFit(PosC, iHistPosC, LEBottomHistValueC);



%% Figure properties
BasePathS = 'C:\E\Dropbox\Lab\05_Scaling\Plots\inferProfiles\IntroducingScaling';
if ~exist(BasePathS, 'dir')
    mkdir(BasePathS);
end
DateS = datestr(datenum(date, 'dd-mmm-yyyy'), 'yyyy-mm-dd');
SavePathS = [BasePathS, filesep, DateS];
if ~exist(SavePathS, 'dir')
    mkdir(SavePathS);
end

FontSize = 14;
%LegendFontSize = 12;
LetterFontSize = 21;
AxesHeight = 8.25; % In centimeters
SmallAxesHeight = 2; % In centimeters
SmallAxesWidth = 2; % In centimeters
AxesWidth = 8.25; % In centimeters
Margin = 2.5; % In centimeters
SmallMargin = 1.25; % In centimeters
LetterPositionR = [-0.20, 0.97];

LineWidth = 1;

SEColorR = [0, 0.75, 0.75];
LEColorR = [0.75, 0, 0.75];
MainGuideLineColorR = [0, 0, 0];
MainGuideLineStyleS = '--';
SEGuideLineColorR = [0, 1, 1];
LEGuideLineColorR = [1, 0, 1];
SEGuideLineStyleSRCA = {'--', '--', '--'};
LEGuideLineStyleSRCA = {'--', '--', '--'};
SEHistColorR = [0.00, 0.75, 0.75];
LEHistColorR = [0.75, 0.00, 0.75];

ErrorStep = 25;
iErrorPosC = (51:ErrorStep:951)';

FigWidth = 2*Margin + SmallMargin + AxesWidth + SmallAxesWidth;
FigHeight = 2*Margin + SmallMargin + AxesHeight + SmallAxesHeight;
Fig = figure('PaperUnits', 'centimeters',...
    'PaperPosition', [0 0 FigWidth FigHeight],...
    'PaperPositionMode', 'manual',...
    'PaperSize', [FigWidth FigHeight],...
    'Units', 'centimeters',...
    'Position', [0 0 FigWidth FigHeight],...
    'WindowStyle', 'docked');



%% SubB

axes('Units', 'centimeters', 'FontSize', FontSize,...
    'Position', [Margin+SmallMargin+SmallAxesWidth, Margin+SmallMargin+SmallAxesHeight, AxesWidth, AxesHeight]);
hold all;
%text('Units', 'normalized', 'Position', LetterPositionR, 'String', 'B', 'FontSize', LetterFontSize);

plot(PosC(iPosRangeC), MeanSEProfileC(iPosRangeC), 'Color', SEColorR);
PlotHandle =...
    errorbar(PosC(iErrorPosC), MeanSEProfileC(iErrorPosC), SD_SEProfileC(iErrorPosC),...
    'Color', SEColorR, 'LineStyle', 'none');
errorbar_tick(PlotHandle, 0);
plot(PosC(iPosRangeC), MeanLEProfileC(iPosRangeC), 'Color', LEColorR);
PlotHandle =...
    errorbar(PosC(iErrorPosC), MeanLEProfileC(iErrorPosC), SD_LEProfileC(iErrorPosC),...
    'Color', LEColorR, 'LineStyle', 'none');
errorbar_tick(PlotHandle, 0);

% Guide lines and patches
%{
line([0, max([SEBottomGuideLineR, LEBottomGuideLineR])], [BcdConcThreshold, BcdConcThreshold],...
    'Color', MainGuideLineColorR, 'LineStyle', MainGuideLineStyleS, 'LineWidth', LineWidth);
%}
%{
ErrorPatch = patch([SEBottomGuideLineR(1), SEBottomGuideLineR(3), SEBottomGuideLineR(3), SEBottomGuideLineR(1)],...
    [0, 0, BcdConcThreshold, BcdConcThreshold],...
    [0, 0, 0]);
set(ErrorPatch, 'EdgeColor', 'none', 'FaceColor', SEGuideLineColorR);
ErrorPatch = patch([LEBottomGuideLineR(1), LEBottomGuideLineR(3), LEBottomGuideLineR(3), LEBottomGuideLineR(1)],...
    [0, 0, BcdConcThreshold, BcdConcThreshold],...
    [0, 0, 0]);
set(ErrorPatch, 'EdgeColor', 'none', 'FaceColor', LEGuideLineColorR);
%}
line([SEBottomGuideLineR(2), SEBottomGuideLineR(2)], [0, BcdConcThreshold],...
    'Color', SEGuideLineColorR, 'LineStyle', SEGuideLineStyleSRCA{2}, 'LineWidth', LineWidth);
line([LEBottomGuideLineR(2), LEBottomGuideLineR(2)], [0, BcdConcThreshold],...
    'Color', LEGuideLineColorR, 'LineStyle', LEGuideLineStyleSRCA{2}, 'LineWidth', LineWidth);
%{
for lLine = 1:length(SEBottomGuideLineR)
    line([SEBottomGuideLineR(lLine), SEBottomGuideLineR(lLine)], [0, BcdConcThreshold],...
        'Color', SEGuideLineColorR, 'LineStyle', SEGuideLineStyleSRCA{lLine}, 'LineWidth', LineWidth);
end
for lLine = 1:length(LEBottomGuideLineR)
    line([LEBottomGuideLineR(lLine), LEBottomGuideLineR(lLine)], [0, BcdConcThreshold],...
        'Color', LEGuideLineColorR, 'LineStyle', LEGuideLineStyleSRCA{lLine}, 'LineWidth', LineWidth);
end
%}

%{
for lEm = 1:length(EmLengthR)
    if SEFilterLR(lEm)
        plot(PosC(iPosRangeC), NormProfileM(iPosRangeC, lEm, :), 'Color', SEColorR);
    else
        plot(PosC(iPosRangeC), NormProfileM(iPosRangeC, lEm, :), 'Color', LEColorR);
    end
end
%}
    
line([0, HistogramAPPos], [SELeftGuideLineR(2), SELeftGuideLineR(2)],...
    'Color', SEGuideLineColorR, 'LineStyle', SEGuideLineStyleSRCA{2}, 'LineWidth', LineWidth);
line([0, HistogramAPPos], [LELeftGuideLineR(2), LELeftGuideLineR(2)],...
    'Color', LEGuideLineColorR, 'LineStyle', LEGuideLineStyleSRCA{2}, 'LineWidth', LineWidth);

%xlabel('AP position (x/L)');
set(gca, 'XTick', 0:0.2:1);
set(gca, 'XTickLabel', []);
xlim([0, 1]);
ylim([0, 1.25]);
set(gca, 'YTick', 0:0.2:1.2);
set(gca, 'YTickLabel', []);
box on;



%% SubA

axes('Units', 'centimeters', 'FontSize', FontSize,...
    'Position', [Margin, Margin+SmallMargin+SmallAxesHeight, SmallAxesWidth, AxesHeight]);
hold all;
%text('Units', 'normalized', 'Position', LetterPositionR, 'String', 'A', 'FontSize', LetterFontSize);

BarOffset = 0.25 * mean(diff(FluorescenceC(iLeftHistPosC)));
XLimR = [0, 1.2*max([SELeftHistValueC; LELeftHistValueC])];
YLimR = [0, 1.25];
SEBinC = FluorescenceC(iLeftHistPosC)-BarOffset;
SEValueC = OnlyPositive(SELeftHistValueC);
barh(SEBinC(~isnan(SEValueC)), SEValueC(~isnan(SEValueC)),...
    0.5, 'FaceColor', SEHistColorR, 'EdgeColor', 'none');
LEBinC = FluorescenceC(iLeftHistPosC)+BarOffset;
LEValueC = OnlyPositive(LELeftHistValueC);
barh(LEBinC(~isnan(LEValueC)), LEValueC(~isnan(LEValueC)),...
    0.5, 'FaceColor', LEHistColorR, 'EdgeColor', 'none');
plot(SELeftGaussC, FluorescenceC, 'Color', SEGuideLineColorR, 'LineWidth', LineWidth);
plot(LELeftGaussC, FluorescenceC, 'Color', LEGuideLineColorR, 'LineWidth', LineWidth);

xlabel('P(m|x)');
xlim(XLimR);
ylabel('Fluorescence (au)');
ylim(YLimR);
set(gca, 'YTick', 0:0.2:1.2);
box on;



%% SubC

axes('Units', 'centimeters', 'FontSize', FontSize,...
    'Position', [Margin+SmallMargin+SmallAxesWidth, Margin, AxesWidth, SmallAxesHeight]);
hold all;
%text('Units', 'normalized', 'Position', LetterPositionR, 'String', 'C', 'FontSize', LetterFontSize);

BarOffset = 0.25 * mean(diff(PosC(iHistPosC)));
XLimR = [0, 1];
YLimR = [0, 1.2*max([SEBottomHistValueC; LEBottomHistValueC])];
SEBinC = PosC(iHistPosC)-BarOffset;
SEValueC = OnlyPositive(SEBottomHistValueC);
bar(SEBinC(~isnan(SEValueC)), SEValueC(~isnan(SEValueC)),...
    0.5, 'FaceColor', SEHistColorR, 'EdgeColor', 'none');
LEBinC = PosC(iHistPosC)+BarOffset;
LEValueC = OnlyPositive(LEBottomHistValueC);
bar(LEBinC(~isnan(LEValueC)), LEValueC(~isnan(LEValueC)),...
    0.5, 'FaceColor', LEHistColorR, 'EdgeColor', 'none');
plot(PosC, SEBottomGaussC, 'Color', SEGuideLineColorR, 'LineWidth', LineWidth);
plot(PosC, LEBottomGaussC, 'Color', LEGuideLineColorR, 'LineWidth', LineWidth);

xlabel('AP position (x/L)');
xlim(XLimR);
set(gca, 'XTick', 0:0.2:1);
ylabel('P(x|m)');
ylim(YLimR);
box on;



%% Saving
FigNameS = sprintf('IntroducingScaling_%s_%s_%d', DataSetS, LengthBinMethodS, iHistogramAPPos);
saveas(gcf, [SavePathS, filesep, FigNameS, '.fig']);
print('-dpdf', '-r600', [SavePathS, filesep, FigNameS, '.pdf']);
print('-dpng', '-r600', [SavePathS, filesep, FigNameS, '.png']);

end



%% CreateGaussianFit
function FitC = CreateGaussianFit(X_C, iBinC, HistC)

GaussFit = fit(X_C(iBinC), HistC, 'gauss1');
FitC = GaussFit.a1*exp(-((X_C-GaussFit.b1)/GaussFit.c1).^2);
Tolerance = 0.001;
FitC(FitC < Tolerance) = NaN;

end


%% OnlyPositive
function PosC = OnlyPositive(OrigC)

PosC = NaN(size(OrigC));
PosC(OrigC > 0) = OrigC(OrigC > 0);

end