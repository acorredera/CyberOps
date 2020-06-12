import { CUSTOM_ELEMENTS_SCHEMA } from '@angular/core';
import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { MedioAmbientePage } from './medio-ambiente.page';

describe('MedioAmbientePage', () => {
  let component: MedioAmbientePage;
  let fixture: ComponentFixture<MedioAmbientePage>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ MedioAmbientePage ],
      schemas: [CUSTOM_ELEMENTS_SCHEMA],
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(MedioAmbientePage);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
