unit Tests.Fixture;

interface

uses
  DUnitX.TestFramework;

type
  [TestFixture]
  TTestFixture = class
  public
    [Setup]
    procedure Setup; virtual;
    [TearDown]
    procedure TearDown; virtual;
  end;

implementation

procedure TTestFixture.Setup;
begin
end;

procedure TTestFixture.TearDown;
begin
end;

end.
