unit Neural.Layers.Conv;

interface

uses
  Neural,
  Neural.Layers;

type
  TConv2D = class(TNeuralLayer)
  private
    FKernels: TArray<TNums>;
    FLastInput: TArray<Single>;
    FLastOutput: TArray<Single>;
    procedure ApplyKernel(X, Y: Integer; const Input, Target: TNums);
    procedure ApplySingleKernel(X, Y, Index: Integer; const Input, Target: TNums);
    procedure DeriveKernel(X, Y: Integer; const AKernel, AGradients, AInput: TNums);
  public
    procedure Build; override;
    function FeedForward(const Input: TArray<Single>): TArray<Single>; override;
    function Backpropagade(const AGradients: TArray<Single>; const ALearningRate: Single): TArray<Single>; override;
    procedure DrawFeatureMap(AIndex: Integer; const AInput, ATarget: TNums);
  end;


implementation

{ TConv2D }

procedure TConv2D.ApplyKernel(X, Y: Integer; const Input, Target: TNums);
var
  LVal: Single;
  i, k, f: Integer;
begin
  LVal := 0;
  for f := 0 to High(FKernels) do
  begin
    for i := 0 to Pred(FKernels[f].Shape.Width) do
    begin
      for k := 0 to Pred(FKernels[f].Shape.Height)  do
        LVal := LVal + (Input[i+X, k+Y] * FKernels[f][i, k]);
    end;
  end;
  Target[X, Y] := LVal;
end;

procedure TConv2D.ApplySingleKernel(X, Y, Index: Integer; const Input,
  Target: TNums);
var
  LVal: Single;
  i, k: Integer;
begin
  LVal := 0;
  for i := 0 to Pred(FKernels[Index].Shape.Width) do
  begin
    for k := 0 to Pred(FKernels[Index].Shape.Height)  do
      LVal := LVal + (Input[i+X, k+Y] * FKernels[Index][i, k]);
  end;
  Target[X, Y] := LVal;
end;
function TConv2D.Backpropagade(const AGradients: TArray<Single>;
  const ALearningRate: Single): TArray<Single>;
var
  LDKernel: TNums;
  LInput: TNums;
  LGradients: TNums;
  i, k, f: Integer;
begin
  for f := 0 to High(FKernels) do
  begin
    LDKernel := TNums.Create(FKernels[f].Shape);
    LInput := TNums.Create(FInputShape, FLastInput);
    LGradients := TNums.Create(FOutputShape, AGradients);
    for i := 0 to Pred(LGradients.Shape.Width) do
      for k := 0 to Pred(LGradients.Shape.Height) do
        DeriveKernel(i, k, LDKernel, LGradients, LInput);

    for i := 0 to Pred(LDKernel.Shape.Width) do
      for k := 0 to Pred(LDKernel.Shape.Height) do
        FKernels[f][i, k] := FKernels[f][i, k] - ALearningRate * LDKernel[i, k];
  end;
end;

procedure TConv2D.Build;
var
  i: Integer;
begin
  inherited;
  SetLength(FKernels, 32);
  for i := Low(FKernels) to High(FKernels) do
    FKernels[i] := FWeightInitializer([3, 3]);
  FOutputShape := [FInputShape.Width - 2, FInputShape.Height - 2];
end;

procedure TConv2D.DeriveKernel(X, Y: Integer; const AKernel, AGradients, AInput: TNums);
var
  i, k: Integer;
begin
  for i := 0 to Pred(AKernel.Shape.Width) do
    for k := 0 to Pred(AKernel.Shape.Height) do
      AKernel[i, k] := AKernel[i, k] + (AGradients[X, Y] * AInput[i+X, k+Y]);
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
  i, k: Integer;
begin
  FLastInput := Input;
  LInput := TNums.Create(FInputShape, Input);
  LOutput := TNums.Create(FOutputShape);
  for i := 0 to Pred(LOutput.Shape.Width) do
  begin
    for k := 0 to Pred(LOutput.Shape.Height) do
      ApplyKernel(i, k, LInput, LOutput);
  end;
  Result := LOutput.Flat;
  FLastOutput := Result;
end;

end.
