import { CUSTOM_ELEMENTS_SCHEMA } from '@angular/core';
import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { DataCenterPage } from './data-center.page';

describe('DataCenterPage', () => {
  let component: DataCenterPage;
  let fixture: ComponentFixture<DataCenterPage>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ DataCenterPage ],
      schemas: [CUSTOM_ELEMENTS_SCHEMA],
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(DataCenterPage);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
