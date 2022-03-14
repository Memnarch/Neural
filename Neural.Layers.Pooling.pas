unit Neural.Layers.Pooling;

interface

uses
  System.Types,
  Neural,
  Neural.Layers;

type
  TMaxPoolingLayer = class(TNeuralLayer)
  private
    FSize: TSize;
    FLastInput: TArray<Single>;
    FLastOutput: TArray<Single>;
    procedure Sample(X, Y, Z: Integer; const AInput, AOutput: TNums);
    procedure PutBacki(X, Y, Z: Integer; const AInputs, AOutput: TNums; const AHighValue, AGradient: Single);
  public
    constructor Create(const ASize: TSize); reintroduce;
    procedure Build; override;
    function FeedForward(const Input: TArray<Single>): TArray<Single>; override;
    function Backpropagade(const AGradients: TArray<Single>; const ALearningRate: Single): TArray<System.Single>; override;
  end;

implementation

{$EXCESSPRECISION OFF}

{ TMaxPoolingLayer }

function TMaxPoolingLayer.Backpropagade(const AGradients: TArray<Single>;
  const ALearningRate: Single): TArray<System.Single>;
var
  i, k, m: Integer;
  LInput, LOutput, LLastOutput, LGradients: TNums;
begin
  LInput := TNums.Create(FInputShape, FLastInput);
  LOutput := TNums.Create(FInputShape);
  LLastOutput := TNums.Create(FOutputShape, FLastOutput);
  LGradients := TNums.Create(FOutputShape, AGradients);

  for i := 0 to Pred(FOutputShape.Width) do
  begin
    for k := 0 to Pred(FOutputShape.Height) do
      for m := 0 to Pred(FOutputShape.Depth) do
        PutBacki(i * FSize.cx, k * FSize.cy, m, LInput, LOutput, LLastOutput[i, k, m], LGradients[i, k, m]);
  end;
  Result := LOutput.Flat;
end;

procedure TMaxPoolingLayer.Build;
begin
  inherited;
  FOutputShape := Copy(FInputShape);
  FOutputShape[0] := FInputShape[0] div FSize.cx;
  FOutputShape[1] := FInputShape[1] div FSize.cy;
end;

constructor TMaxPoolingLayer.Create(const ASize: TSize);
begin
  inherited Create(TActivation.Identity, nil);
  FSize := ASize;
end;

function TMaxPoolingLayer.FeedForward(
  const Input: TArray<Single>): TArray<Single>;
var
  i, k, m: Integer;
  LInput, LOutput: TNums;
begin
  LInput := TNums.Create(FInputShape, Input);
  LOutput := TNums.Create(FOutputShape);
  FLastInput := Input;
  for i := 0 to Pred(LOutput.Shape.Width) do
  begin
    for k := 0 to Pred(LOutput.Shape.Height) do
      for m := 0 to Pred(LOutput.Shape.Depth) do
        Sample(i, k, m, LInput, LOutput);
  end;
  Result := LOutput.Flat;
  FLastOutput := Result;
end;

procedure TMaxPoolingLayer.PutBacki(X, Y, Z: Integer; const AInputs, AOutput: TNums; const AHighValue, AGradient: Single);
var
  i, k: Integer;
begin
  for i := X to X + Pred(FSize.cx) do
  begin
    for k := Y to Y +Pred(FSize.cy) do
    begin
      if AInputs[i, k, z] <> AHighValue then
        AOutput[i, k, Z] := 0
      else
        AOutput[i, k, Z] := AGradient;
    end;
  end;
end;

procedure TMaxPoolingLayer.Sample(X, Y, Z: Integer; const AInput, AOutput: TNums);
var
  i, k, LX, LY: Integer;
  LVal, LCurrent: Single;
begin
  LX := X*FSize.cx;
  LY := Y*FSize.cy;
  LVal := AInput[LX, LY, Z];
  for i := 0 to Pred(FSize.cx) do
  begin
    for k := 0 to Pred(FSize.cy) do
    begin
      LCurrent := AInput[LX + i, LY + k, Z];
      if LCurrent > LVal then
        LVal := LCurrent;
    end;
  end;
  AOutput[X, Y, Z] := LVal;
end;

end.
