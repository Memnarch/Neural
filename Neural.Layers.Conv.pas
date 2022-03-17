unit Neural.Layers.Conv;

interface

uses
  Neural,
  Neural.Layers;

type
  TConv2D = class(TNeuralLayer)
  private
    FKernelSize: TShape;
    FFilterCount: Integer;
    FFilters: TArray<TArray<TNums>>;
    FLastInput: TNums;
    FLastOutput: TNums;
    FLastBackpropagade: TNums;
    procedure ApplyKernel(X, Y, Z: Integer; const Input, Target: PNums);
    procedure ApplySingleKernel(X, Y, Index: Integer; const Input, Target: TNums);
    procedure DeriveKernel(X, Y, Z: Integer; AGradient: Single; const AOrigKernel: TNums; const AKernel, AInput, ATargetGradients: TNums);
    procedure RaiseInvalidShape;
  public
    constructor Create(AFilters: Integer; AKernelSize: TShape; const AActivation: TActivation; const AWeightInitializer: TInitializerFunc = nil); reintroduce;
    procedure Build; override;
    function FeedForward(const Input: TArray<Single>): TArray<Single>; override;
    function Backpropagade(const AGradients: TArray<Single>; const ALearningRate: Single): TArray<Single>; override;
    procedure DrawFeatureMap(AIndex: Integer; const AInput, ATarget: TNums);
  end;


implementation

uses
  SysUtils;

{ TConv2D }

{$ExcessPrecision OFF}
{$PointerMath ON }

procedure TConv2D.ApplyKernel(X, Y, Z: Integer; const Input, Target: PNums);
var
  LVal: Single;
  i, k, f: Integer;
  LKernel: PNums;
  LInputRef, LKernelRef: PSingle;
begin
  LVal := 0;
  for f := 0 to High(FFilters) do
  begin
    LKernel := @FFilters[f][Z];
    for k := 0 to Pred(FKernelSize.Height)  do
    begin
      LInputRef := Input.Ref(X, k+Y, Z);
      LKernelRef := LKernel.Ref(0, k);
      for i := 0 to Pred(FKernelSize.Width) do
        LVal := LVal + (LInputRef[i] * LKernelRef[i]);
    end;
  end;
  Target^[X, Y, Z] := FActivation.Run(LVal);
end;

procedure TConv2D.ApplySingleKernel(X, Y, Index: Integer; const Input, Target: TNums);
var
  LVals: TNums;
  i, k, m: Integer;
begin
  LVals := TNums.Create([InputShape.Depth]);
  for i := 0 to Pred(FKernelSize.Width) do
  begin
    for k := 0 to Pred(FKernelSize.Height) do
      for m := 0 to Pred(FInputShape.Depth) do
        LVals[m] := LVals[m] + (Input[i+X, k+Y, m] * FFilters[Index][m][i, k]);
  end;
  for m := 0 to Pred(LVals.Shape.Width) do
    Target[X, Y, m] := LVals[m];
end;

function TConv2D.Backpropagade(const AGradients: TArray<Single>;
  const ALearningRate: Single): TArray<Single>;
var
  LDKernel: TNums;
  LGradients: TNums;
  i, k, m, f: Integer;
  LFilter: TArray<TNums>;
  LKernel: PNums;
begin
  FLastBackpropagade.FillZero;
  LDKernel := TNums.Create(FKernelSize + [FInputShape.Depth]);
  LGradients := TNums.Create(FOutputShape, AGradients);
  for f := 0 to High(FFilters) do
  begin
    LDKernel.FillZero;
    LFilter := FFilters[f];
    for i := 0 to Pred(LGradients.Shape.Width) do
      for k := 0 to Pred(LGradients.Shape.Height) do
        for m := 0 to Pred(LGradients.Shape.Depth) do
          DeriveKernel(i, k, m, LGradients[i, k, m], LFilter[m], LDKernel, FLastInput, FLastBackpropagade);

    for m := 0 to Pred(LDKernel.Shape.Depth) do
    begin
      LKernel := @LFilter[m];
      for i := 0 to Pred(LDKernel.Shape.Width) do
        for k := 0 to Pred(LDKernel.Shape.Height) do
            LKernel^[i, k] := LKernel^[i, k] - ALearningRate * LDKernel[i, k, m];
    end;
  end;
  Result := FLastBackpropagade.Flat;
end;

procedure TConv2D.Build;
var
  i, k: Integer;
begin
  inherited;
  if Length(FKernelSize) <> 2 then
    RaiseInvalidShape;
  SetLength(FFilters, FFilterCount);
  for i := Low(FFilters) to High(FFilters) do
  begin
    SetLength(FFilters[i], FInputShape.Depth);
    for k := 0 to Pred(FInputShape.Depth) do
      FFilters[i][k] := FWeightInitializer(FKernelSize);
  end;
  FOutputShape := Copy(FInputShape);
  FOutputShape[0] := FInputShape[0] - Pred(FKernelSize[0]);
  FOutputShape[1] := FInputShape[1] - Pred(FKernelSize[0]);
  FLastOutput.Reshape(FOutputShape);
  FLastBackpropagade.Reshape(FInputShape);
end;

constructor TConv2D.Create(AFilters: Integer; AKernelSize: TShape;
  const AActivation: TActivation; const AWeightInitializer: TInitializerFunc);
begin
  inherited Create(AActivation, AWeightInitializer);
  FFilterCount := AFilters;
  FKernelSize := AKernelSize;
end;

procedure TConv2D.DeriveKernel(X, Y, Z: Integer; AGradient: Single; const AOrigKernel: TNums; const AKernel, AInput, ATargetGradients: TNums);
var
  i, k: Integer;
  LDActivated, LGradient: Single;
  LInputRef, LTargetRef, LOrigKernelRef, LKernelRef: PSingle;
begin
  for k := 0 to Pred(AKernel.Shape.Height) do
  begin
    LInputRef := AInput.Ref(X, k+Y, Z);
    LTargetRef := ATargetGradients.Ref(X, Y+k, Z);
    LOrigKernelRef := AOrigKernel.Ref(0, k);
    LKernelRef := AKernel.Ref(0, k, Z);
    for i := 0 to Pred(AKernel.Shape.Width) do
    begin
      LDActivated := FActivation.Derive(LInputRef[i]);
      LGradient := AGradient * LDActivated;
      LTargetRef[i] := LTargetRef[i] + LOrigKernelRef[i] * LGradient;
      LKernelRef[i] := LKernelRef[i] + LGradient;
    end;
  end;
end;

procedure TConv2D.DrawFeatureMap(AIndex: Integer; const AInput, ATarget: TNums);
var
  i, k: Integer;
begin
  for i := 0 to Pred(ATarget.Shape.Width) do
  begin
    for k := 0 to Pred(Atarget.Shape.Height) do
      ApplySingleKernel(i, k, AIndex, AInput, ATarget);
  end;
end;

function TConv2D.FeedForward(const Input: TArray<Single>): TArray<Single>;
var
  i, k, m: Integer;
begin
  FLastOutput.FillZero;
  FLastInput := TNums.Create(FInputShape, Input);
  for i := 0 to Pred(FLastOutput.Shape.Width) do
  begin
    for k := 0 to Pred(FLastOutput.Shape.Height) do
      for m := 0 to Pred(FLastOutput.Shape.Depth) do
        ApplyKernel(i, k, m, @FLastInput, @FLastOutput);
  end;
  Result := FLastOutput.Flat;
end;

procedure TConv2D.RaiseInvalidShape;
begin
  raise Exception.Create('KernelShape must have 2 Dimensions');
end;

end.
