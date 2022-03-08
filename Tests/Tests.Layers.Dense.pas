unit Tests.Layers.Dense;

interface

uses
  DUnitX.TestFramework,
  Neural.Layers,
  Tests.Layers;

type
  TDenseLayerTests = class(TNeuralLayerTest)
  protected
      function BuildLayer(Index: Integer; const ARun: TTestRun): TNeuralLayer; override;
      function GetRuns: TArray<TTestRun>; override;
  end;

implementation

uses
  Neural,
  Neural.Layers.Dense;

{ TDenseLayerTests }

function TDenseLayerTests.BuildLayer(Index: Integer; const ARun: TTestRun): TNeuralLayer;
begin
  Result := TDenseLayer.Create(2, TActivation.Identity, TInitializers.Fixed([0.5, 0.25]));
end;

function TDenseLayerTests.GetRuns: TArray<TTestRun>;
var
  LRun: TTestRun;
begin
  LRun.Input := TNums.Create([2], [8, 20]);
  LRun.Output := TNums.Create([2],[9, 9]);
  LRun.InGradients := TNums.Create([2], [4, 5]);
  LRun.OutGradients := TNums.Create([2], [40.5, 20.25]);
  Result := [LRun];
end;

initialization
  TDUnitX.RegisterTestFixture(TDenseLayerTests);

end.
