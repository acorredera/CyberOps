import { CUSTOM_ELEMENTS_SCHEMA } from '@angular/core';
import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { FinanzasPage } from './finanzas.page';

describe('FinanzasPage', () => {
  let component: FinanzasPage;
  let fixture: ComponentFixture<FinanzasPage>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ FinanzasPage ],
      schemas: [CUSTOM_ELEMENTS_SCHEMA],
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(FinanzasPage);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
