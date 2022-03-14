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
    FLastInput: TArray<Single>;
    FLastOutput: TArray<Single>;
    procedure ApplyKernel(X, Y, Z: Integer; const Input, Target: TNums);
    procedure ApplySingleKernel(X, Y, Index: Integer; const Input, Target: TNums);
    procedure DeriveKernel(X, Y, Z: Integer; const AOrigKernels: TArray<TNums>; const AKernel, AGradients, AInput, ATargetGradients: TNums);
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

procedure TConv2D.ApplyKernel(X, Y, Z: Integer; const Input, Target: TNums);
var
  LVal: Single;
  i, k, f: Integer;
begin
  LVal := 0;
  for f := 0 to High(FFilters) do
  begin
    for i := 0 to Pred(FKernelSize.Width) do
    begin
      for k := 0 to Pred(FKernelSize.Height)  do
          LVal := LVal + (Input[i+X, k+Y, Z] * FFilters[f][z][i, k]);
    end;
  end;
  Target[X, Y, Z] := FActivation.Run(LVal);
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
  LInput: TNums;
  LGradients: TNums;
  i, k, m, f: Integer;
  LResult: TNums;
begin
  LResult := TNums.Create(FInputShape);
  for f := 0 to High(FFilters) do
  begin
    LDKernel := TNums.Create(FKernelSize + [FInputShape.Depth]);
    LInput := TNums.Create(FInputShape, FLastInput);
    LGradients := TNums.Create(FOutputShape, AGradients);
    for i := 0 to Pred(LGradients.Shape.Width) do
      for k := 0 to Pred(LGradients.Shape.Height) do
        for m := 0 to Pred(LGradients.Shape.Depth) do
          DeriveKernel(i, k, m, FFilters[f], LDKernel, LGradients, LInput, LResult);

    for i := 0 to Pred(LDKernel.Shape.Width) do
      for k := 0 to Pred(LDKernel.Shape.Height) do
        for m := 0 to Pred(LDKernel.Shape.Depth) do
          FFilters[f][m][i, k] := FFilters[f][m][i, k] - ALearningRate * LDKernel[i, k, m];
  end;
  Result := LResult.Flat;
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
end;

constructor TConv2D.Create(AFilters: Integer; AKernelSize: TShape;
  const AActivation: TActivation; const AWeightInitializer: TInitializerFunc);
begin
  inherited Create(AActivation, AWeightInitializer);
  FFilterCount := AFilters;
  FKernelSize := AKernelSize;
end;

procedure TConv2D.DeriveKernel(X, Y, Z: Integer; const AOrigKernels: TArray<TNums>; const AKernel, AGradients, AInput, ATargetGradients: TNums);
var
  i, k: Integer;
  LDActivated: Single;
begin
  for i := 0 to Pred(AKernel.Shape.Width) do
    for k := 0 to Pred(AKernel.Shape.Height) do
    begin
      LDActivated := FActivation.Derive(AInput[i+X, k+Y, Z]);
      ATargetGradients[x+i, y+k, Z] := ATargetGradients[x+i, y+k, Z] + AOrigKernels[Z][i, k] * LDActivated * AGradients[X, Y, Z];
      AKernel[i, k, Z] := AKernel[i, k, Z] + (AGradients[X, Y, Z] * LDActivated);
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
  LInput: TNums;
  LOutput: TNums;
  i, k, m: Integer;
begin
  FLastInput := Input;
  LInput := TNums.Create(FInputShape, Input);
  LOutput := TNums.Create(FOutputShape);
  for i := 0 to Pred(LOutput.Shape.Width) do
  begin
    for k := 0 to Pred(LOutput.Shape.Height) do
      for m := 0 to Pred(LOutput.Shape.Depth) do
        ApplyKernel(i, k, m, LInput, LOutput);
  end;
  Result := LOutput.Flat;
  FLastOutput := Result;
end;

procedure TConv2D.RaiseInvalidShape;
begin
  raise Exception.Create('KernelShape must have 2 Dimensions');
end;

end.
