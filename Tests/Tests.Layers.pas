unit Tests.Layers;

interface

uses
  DUnitX.TestFramework,
  Tests.Fixture,
  Neural,
  Neural.Layers;

type
  TTestRun = record
    Input: TNums;
    Output: TNums;
    InGradients: TNums;
    OutGradients: TNums;
  end;

  TNeuralLayerTest = class(TTestFixture)
  protected
    FSut: TNeuralLayer;
    function BuildLayer(Index: Integer; const ARun: TTestRun): TNeuralLayer; virtual; abstract;
    function GetRuns: TArray<TTestRun>; virtual; abstract;
    procedure CompareNums(const AExpected, ANum: TNums);
  public
    procedure Setup; override;
    procedure TearDown; override;
    [Test]
    procedure TestLayer;
  end;

implementation

uses
  Math,
  SysUtils;

procedure TNeuralLayerTest.CompareNums(const AExpected, ANum: TNums);
var
  i: Integer;
begin
  Assert.AreEqual(Length(AExpected.Shape), Length(ANum.Shape), 'Shape.Length');
  Assert.AreEqual(AExpected.Shape.Size, ANum.Shape.Size, 'Shape.Size');
  for i := 0 to Pred(AExpected.Shape.Size) do
  begin
    Assert.AreEqual(AExpected.Flat[i], ANum.Flat[i], 'Value missmatch at index ' + IntToStr(i));
  end;
end;

procedure TNeuralLayerTest.Setup;
begin

end;

procedure TNeuralLayerTest.TearDown;
begin

end;


procedure TNeuralLayerTest.TestLayer;
var
  LLayer: TNeuralLayer;
  LRun: TTestRun;
  LActualOutput, LActualGradients: TNums;
  LRuns: TArray<TTestRun>;
  i: Integer;
begin
  LRuns := GetRuns();
  for i := Low(LRuns) to High(LRuns) do
  begin
    LRun := LRuns[i];
    LLayer := BuildLayer(i, LRun);
    try
      LLayer.InputShape := LRun.Input.Shape;
      LLayer.Build;
      LActualOutput := TNums.Create(LLayer.OutputShape, LLayer.FeedForward(LRun.Input.Flat));
      CompareNums(LRun.Output, LActualOutput);
      LActualGradients := TNums.Create(LLayer.InputShape, LLayer.Backpropagade(LRun.InGradients.Flat, 0.1));
      CompareNums(LRun.OutGradients, LActualGradients);
    finally
      LLayer.Free;
    end;
  end;
end;

end.
