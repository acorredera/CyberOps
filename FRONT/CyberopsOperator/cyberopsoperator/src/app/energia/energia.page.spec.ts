import { CUSTOM_ELEMENTS_SCHEMA } from '@angular/core';
import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { EnergiaPage } from './energia.page';

describe('EnergiaPage', () => {
  let component: EnergiaPage;
  let fixture: ComponentFixture<EnergiaPage>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ EnergiaPage ],
      schemas: [CUSTOM_ELEMENTS_SCHEMA],
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(EnergiaPage);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
